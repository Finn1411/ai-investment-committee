"""
Committee Engine — Week 4 core (built early as it's the heart of the system).

Orchestrates all agents in the correct sequence:
  1. Fundamental Analyst
  2. Value Analyst
  3. Growth Analyst
  4. Earnings & Catalyst Analyst
  5. Risk Manager
  6. Bear Analyst (receives bull summary to refute)
  7. Committee Engine (weighted scoring → CommitteeVerdict)
  8. Portfolio Manager (final fit check + position size)

The Committee uses a transparent, deterministic scoring system.
Weights come from config.yaml and can be backtested later.

Output: CommitteeVerdict (BUY / HOLD / AVOID) with full audit trail.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import google.genai as genai
from google.genai import types as genai_types

from finance_agent.agents.bear_analyst import BearAnalyst
from finance_agent.agents.earnings_analyst import EarningsAnalyst
from finance_agent.agents.fundamental_analyst import FundamentalAnalyst
from finance_agent.agents.growth_analyst import GrowthAnalyst
from finance_agent.agents.portfolio_manager import PortfolioContext, PortfolioManager
from finance_agent.agents.risk_manager import RiskManager
from finance_agent.agents.value_analyst import ValueAnalyst
from finance_agent.data.pipeline import AnalysisContext
from finance_agent.database.db import AgentAnalysisORM, init_db, get_session
from finance_agent.evaluation.journal import PredictionJournal
from finance_agent.models.schemas import AgentAnalysis, CommitteeVerdict, Horizon, Rating
from finance_agent.utils.config import settings
from finance_agent.utils.logger import logger


# ── Committee weights (from masterplan) ──────────────────────────────────────
# Sum must equal 1.0. Override in config.yaml later.
DEFAULT_WEIGHTS = {
    "FundamentalAnalyst": 0.20,   # Business quality
    "ValueAnalyst":        0.20,  # Valuation / margin of safety
    "GrowthAnalyst":       0.15,  # Growth quality
    "EarningsAnalyst":     0.10,  # Catalysts
    "RiskManager":         0.15,  # Risk (high risk → lower score)
    "BearAnalyst":         0.20,  # Bear case strength (high score = weak bear = more bullish)
    # PortfolioManager is advisory only — does not affect committee score
}

# Thresholds for BUY / HOLD / AVOID
BUY_THRESHOLD = 6.5
AVOID_THRESHOLD = 4.0


@dataclass
class CommitteeResult:
    """Full output of the Committee process."""
    ticker: str
    verdict: CommitteeVerdict
    agent_analyses: dict[str, AgentAnalysis]
    portfolio_analysis: Optional[AgentAnalysis]
    weighted_score: float
    agent_scores: dict[str, float]
    agent_confidences: dict[str, float]
    total_runtime_seconds: float = 0.0


class CommitteeEngine:
    """
    Runs all agents in sequence and produces a final CommitteeVerdict.

    Usage:
        result = CommitteeEngine().run(context)
        print(result.verdict.rating)
        print(result.verdict.thesis)
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        portfolio: Optional[PortfolioContext] = None,
        persist_to_db: bool = True,
        log_to_journal: bool = True,
    ) -> None:
        self.weights = weights or DEFAULT_WEIGHTS
        self.portfolio = portfolio or PortfolioContext()
        self.persist_to_db = persist_to_db
        self.log_to_journal = log_to_journal

        # Validate weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Committee weights must sum to 1.0, got {total:.3f}")

        # Instantiate all agents once (shared Gemini client)
        self._fundamental = FundamentalAnalyst()
        self._value = ValueAnalyst()
        self._growth = GrowthAnalyst()
        self._earnings = EarningsAnalyst()
        self._risk = RiskManager()
        self._bear = BearAnalyst()
        self._portfolio = PortfolioManager()

        # Gemini for thesis synthesis
        api_key = os.getenv("GEMINI_API_KEY", "")
        genai.configure = lambda **_: None  # no-op for genai.Client style
        self._synthesis_client = genai.Client(api_key=api_key)
        self._synthesis_model_name = settings.llm.model
        self._synthesis_config = genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=4096,
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        if persist_to_db:
            init_db()

        logger.info("[Committee] Initialised with all agents")

    def run(
        self,
        context: AnalysisContext,
        horizon: Horizon = Horizon.TWELVE_MONTHS,
    ) -> CommitteeResult:
        """
        Full committee run. Agents execute in sequence.

        Args:
            context: AnalysisContext from DataPipeline.run()
            horizon: Investment horizon for the verdict

        Returns:
            CommitteeResult with verdict and all agent analyses
        """
        start = datetime.utcnow()
        ticker = context.ticker
        logger.info(f"[Committee] ====== Starting committee for {ticker} ======")

        agent_analyses: dict[str, AgentAnalysis] = {}

        # ── 1. Fundamental Analyst ─────────────────────────────────────────────
        logger.info(f"[Committee] Running FundamentalAnalyst...")
        agent_analyses["FundamentalAnalyst"] = self._fundamental.analyse(context)

        # ── 2. Value Analyst ──────────────────────────────────────────────────
        logger.info(f"[Committee] Running ValueAnalyst...")
        agent_analyses["ValueAnalyst"] = self._value.analyse(context)

        # ── 3. Growth Analyst ─────────────────────────────────────────────────
        logger.info(f"[Committee] Running GrowthAnalyst...")
        agent_analyses["GrowthAnalyst"] = self._growth.analyse(context)

        # ── 4. Earnings & Catalyst Analyst ────────────────────────────────────
        logger.info(f"[Committee] Running EarningsAnalyst...")
        agent_analyses["EarningsAnalyst"] = self._earnings.analyse(context)

        # ── 5. Risk Manager ───────────────────────────────────────────────────
        logger.info(f"[Committee] Running RiskManager...")
        agent_analyses["RiskManager"] = self._risk.analyse(context)

        # ── 6. Bear Analyst (receives bull summary) ────────────────────────────
        logger.info(f"[Committee] Running BearAnalyst...")
        bull_summary = self._build_bull_summary(agent_analyses)
        agent_analyses["BearAnalyst"] = self._bear.analyse(context, bull_thesis_summary=bull_summary)

        # ── 7. Compute weighted score ─────────────────────────────────────────
        agent_scores = {name: a.score for name, a in agent_analyses.items() if a.score is not None}
        agent_confidences = {name: a.confidence for name, a in agent_analyses.items() if a.confidence is not None}
        weighted_score = self._compute_weighted_score(agent_scores)
        avg_confidence = sum(agent_confidences.values()) / max(len(agent_confidences), 1)

        logger.info(
            f"[Committee] {ticker} | Weighted score: {weighted_score:.2f}/10 | "
            f"Avg confidence: {avg_confidence:.2f}"
        )

        # ── 8. Rating decision ────────────────────────────────────────────────
        rating = self._score_to_rating(weighted_score)
        logger.info(f"[Committee] {ticker} | Rating: {rating.value}")

        # ── 9. Synthesize thesis & invalidation criteria ───────────────────────
        thesis, invalidation_criteria = self._synthesize_thesis(
            context, agent_analyses, weighted_score, rating
        )

        # ── 10. Disagreement Detection ─────────────────────────────────────────
        disagreement_score, conflicting_agents = self._compute_disagreement(
            agent_scores, weighted_score
        )
        disagreement_flag = disagreement_score > 3.0
        if disagreement_flag:
            logger.warning(
                f"[Committee] {ticker} | HIGH DISAGREEMENT detected "
                f"(score={disagreement_score:.2f}) | "
                f"Conflicting: {', '.join(conflicting_agents)}"
            )

        # ── 11. Extract bull/bear narratives ───────────────────────────────────
        bull_summary, bear_summary = self._extract_narratives(agent_analyses)

        # ── 12. Build CommitteeVerdict ─────────────────────────────────────────
        scenario_model = context.scenarios.model if context.scenarios else None
        if scenario_model is None:
            raise ValueError(f"No scenario model available for {ticker}")

        verdict = CommitteeVerdict(
            ticker=ticker,
            analysis_date=date.today(),
            horizon=horizon,
            rating=rating,
            weighted_score=round(weighted_score, 2),
            confidence=round(avg_confidence, 3),
            scenario_model=scenario_model,
            thesis=thesis,
            invalidation_criteria=invalidation_criteria,
            agent_scores=agent_scores,
            agent_confidences=agent_confidences,
            disagreement_score=round(disagreement_score, 2),
            conflicting_agents=conflicting_agents,
            disagreement_flag=disagreement_flag,
            bull_case_summary=bull_summary,
            bear_case_summary=bear_summary,
        )

        # ── 13. Portfolio Manager ─────────────────────────────────────────────
        logger.info(f"[Committee] Running PortfolioManager...")
        portfolio_analysis = None
        try:
            portfolio_analysis = self._portfolio.analyse(
                context, portfolio=self.portfolio, committee_score=weighted_score
            )
        except Exception as e:
            logger.warning(
                f"[Committee] PortfolioManager failed (advisory — result still valid): {e}"
            )

        # ── 14. Persist ───────────────────────────────────────────────────────
        if self.persist_to_db:
            self._persist_analyses(ticker, agent_analyses, portfolio_analysis)

        if self.log_to_journal:
            self._log_to_journal(context, verdict)

        runtime = (datetime.utcnow() - start).total_seconds()
        logger.info(
            f"[Committee] ====== {ticker} complete in {runtime:.1f}s ====== "
            f"Rating={rating.value} | Score={weighted_score:.2f}"
        )

        return CommitteeResult(
            ticker=ticker,
            verdict=verdict,
            agent_analyses=agent_analyses,
            portfolio_analysis=portfolio_analysis,
            weighted_score=weighted_score,
            agent_scores=agent_scores,
            agent_confidences=agent_confidences,
            total_runtime_seconds=runtime,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_weighted_score(self, agent_scores: dict[str, float]) -> float:
        """
        Compute the weighted committee score.
        Agents with no score are excluded, weights renormalized.
        """
        total_weight = 0.0
        total_score = 0.0

        for agent_name, weight in self.weights.items():
            if agent_name in agent_scores:
                total_score += agent_scores[agent_name] * weight
                total_weight += weight

        if total_weight == 0:
            return 5.0  # Neutral fallback

        return total_score / total_weight

    def _score_to_rating(self, score: float) -> Rating:
        """Convert weighted score to BUY / HOLD / AVOID."""
        if score >= BUY_THRESHOLD:
            return Rating.BUY
        elif score <= AVOID_THRESHOLD:
            return Rating.AVOID
        else:
            return Rating.HOLD

    def _compute_disagreement(
        self, agent_scores: dict[str, float], weighted_score: float
    ) -> tuple[float, list[str]]:
        """
        Compute disagreement score = std dev of agent scores * 2.
        Also identify agents whose score diverges >2.5 pts from committee score.

        Returns:
            (disagreement_score, list_of_conflicting_agent_names)
        """
        import statistics
        scores = list(agent_scores.values())
        if len(scores) < 2:
            return 0.0, []

        std_dev = statistics.stdev(scores)
        disagreement_score = min(std_dev * 2.0, 10.0)

        # Agents whose score is >2.5 pts away from the weighted score
        threshold = 2.5
        conflicting = [
            name for name, score in agent_scores.items()
            if abs(score - weighted_score) > threshold
        ]
        return disagreement_score, conflicting

    def _extract_narratives(
        self, analyses: dict[str, "AgentAnalysis"]
    ) -> tuple[str, str]:
        """
        Extract 1-sentence bull and bear case summaries from agent outputs.
        Bull = average of FundamentalAnalyst + GrowthAnalyst summaries.
        Bear = BearAnalyst summary.
        """
        bull_parts = []
        for name in ["FundamentalAnalyst", "GrowthAnalyst", "ValueAnalyst"]:
            if name in analyses and analyses[name].summary:
                # Split on ". " to avoid splitting on decimal numbers like 56.2%
                first_sentence = analyses[name].summary.split(". ")[0].strip()
                # Remove trailing period if it exists (in case there was only 1 sentence)
                if first_sentence.endswith("."):
                    first_sentence = first_sentence[:-1]
                bull_parts.append(first_sentence)

        bull_summary = ". ".join(bull_parts[:2]) + "." if bull_parts else ""

        bear_summary = ""
        if "BearAnalyst" in analyses and analyses["BearAnalyst"].summary:
            bear_summary = analyses["BearAnalyst"].summary

        return bull_summary, bear_summary

    # ── Thesis synthesis ──────────────────────────────────────────────────────

    def _build_bull_summary(self, analyses: dict[str, AgentAnalysis]) -> str:
        """Build a concise bull thesis from the first 4 agents for the Bear Analyst."""
        lines = []
        for name in ["FundamentalAnalyst", "ValueAnalyst", "GrowthAnalyst", "EarningsAnalyst"]:
            if name in analyses:
                a = analyses[name]
                lines.append(f"{name} (score {a.score}/10): {a.summary}")
                if a.key_findings:
                    lines.append(f"  Key: {a.key_findings[0]}")
        return "\n".join(lines)

    def _synthesize_thesis(
        self,
        context: AnalysisContext,
        analyses: dict[str, AgentAnalysis],
        weighted_score: float,
        rating: Rating,
    ) -> tuple[str, list[str]]:
        """
        Use Gemini to synthesize agent outputs into a coherent investment thesis
        and a list of specific invalidation criteria.
        """
        agents_summary = "\n\n".join([
            f"{name} (score {a.score}/10, confidence {a.confidence}):\n{a.summary}\n"
            f"Key findings: {'; '.join(a.key_findings[:3]) if a.key_findings else 'N/A'}"
            for name, a in analyses.items()
        ])

        d = context.to_prompt_dict()

        prompt = f"""You are synthesizing a final investment research memo for {context.ticker} ({context.company_name}).

COMMITTEE DECISION: {rating.value} | Weighted Score: {weighted_score:.2f}/10

AGENT ANALYSES:
{agents_summary}

KEY METRICS:
- Price: ${d.get('current_price', 'N/A')} | Market Cap: ${d.get('market_cap_bn', 'N/A')}B
- Expected Value (Monte Carlo): {d.get('scenario_expected_value', 'N/A')}
- DCF Margin of Safety: {d.get('dcf_margin_of_safety', 'N/A')}
- Valuation Label: {d.get('valuation_label', 'N/A')}
- Piotroski: {d.get('piotroski_f_score', 'N/A')}/9 | Altman Z: {d.get('altman_z_score', 'N/A')}

Write a professional investment thesis memo (3-4 paragraphs) that:
1. States the committee decision clearly and why
2. Summarizes the BULL case with specific evidence
3. Addresses the BEAR case and why it is or isn't decisive
4. States what would change the rating

Then write exactly 5 SPECIFIC, MEASURABLE invalidation criteria that would cause us to EXIT or DOWNGRADE:
Format: "INVALIDATION: [specific metric] [direction] [threshold] -> [consequence]"
Example: "INVALIDATION: Gross margin falls below 40% for 2 consecutive quarters -> DOWNGRADE to AVOID"

Respond with JSON:
{{
  "thesis": "full thesis text",
  "invalidation_criteria": ["criterion 1", "criterion 2", "criterion 3", "criterion 4", "criterion 5"]
}}"""

        try:
            response = self._synthesis_client.models.generate_content(
                model=self._synthesis_model_name,
                contents=prompt,
                config=self._synthesis_config,
            )
            text = response.text.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return parsed.get("thesis", "Analysis complete."), parsed.get("invalidation_criteria", [])
        except Exception as e:
            logger.warning(f"[Committee] Thesis synthesis failed: {e}")

        # Fallback
        thesis = (
            f"Committee verdict: {rating.value} with score {weighted_score:.2f}/10. "
            f"The analysis reflects a {rating.value} recommendation based on the weighted "
            f"assessment of fundamental quality, valuation, growth, risk, and bear case strength."
        )
        return thesis, [
            "INVALIDATION: Operating margin compression >300bps for 2 consecutive quarters -> REVIEW",
            "INVALIDATION: Net Debt/EBITDA exceeds 4x -> DOWNGRADE",
            f"INVALIDATION: Stock price declines >25% from entry without fundamental deterioration -> ADD",
            "INVALIDATION: Revenue growth decelerates to <5% for 2 quarters -> REVIEW",
            "INVALIDATION: Piotroski F-Score drops below 4 -> DOWNGRADE",
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist_analyses(
        self,
        ticker: str,
        agent_analyses: dict[str, AgentAnalysis],
        portfolio_analysis: Optional[AgentAnalysis],
    ) -> None:
        """Store all agent analyses in the DB."""
        all_analyses = {**agent_analyses}
        if portfolio_analysis:
            all_analyses["PortfolioManager"] = portfolio_analysis

        try:
            with get_session() as session:
                for name, analysis in all_analyses.items():
                    session.add(AgentAnalysisORM(
                        ticker=ticker,
                        agent_name=name,
                        analysis_date=str(date.today()),
                        summary=analysis.summary,
                        key_findings=json.dumps(analysis.key_findings),
                        score=analysis.score,
                        confidence=analysis.confidence,
                        raw_output=analysis.raw_output or "",
                    ))
                session.commit()
            logger.debug(f"[Committee] Persisted {len(all_analyses)} analyses for {ticker}")
        except Exception as e:
            logger.error(f"[Committee] DB persist failed: {e}")

    def _log_to_journal(self, context: AnalysisContext, verdict: CommitteeVerdict) -> None:
        """Log the prediction to the Prediction Journal."""
        try:
            journal = PredictionJournal()
            journal.log(
                ticker=context.ticker,
                analysis_date=date.today(),
                horizon=verdict.horizon,
                rating=verdict.rating,
                weighted_score=verdict.weighted_score,
                confidence=verdict.confidence,
                scenario_model=verdict.scenario_model,
                thesis=verdict.thesis,
                invalidation_criteria=verdict.invalidation_criteria,
                agent_scores=verdict.agent_scores,
                metrics_snapshot=context.to_prompt_dict(),
            )
            logger.info(f"[Committee] Prediction logged to journal for {context.ticker}")
        except Exception as e:
            logger.warning(f"[Committee] Journal logging failed: {e}")

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self, result: CommitteeResult) -> None:
        """Print a formatted committee report to console."""
        v = result.verdict
        sep = "=" * 70

        print(f"\n{sep}")
        print(f"  INVESTMENT COMMITTEE VERDICT: {v.ticker}")
        print(sep)
        print(f"  Rating:          {v.rating.value}")
        print(f"  Weighted Score:  {v.weighted_score:.2f} / 10")
        print(f"  Confidence:      {v.confidence:.1%}")
        print(f"  Horizon:         {v.horizon.value}")
        print(f"  Analysis Date:   {v.analysis_date}")
        print(f"\n-- AGENT SCORES --")
        for agent, score in sorted(result.agent_scores.items(), key=lambda x: -x[1]):
            weight = self.weights.get(agent, 0)
            print(f"  {agent:<25} {score:>5.2f}/10  (weight {weight:.0%})")
        print(f"\n-- SCENARIOS --")
        sm = v.scenario_model
        print(f"  Bear ({sm.bear.probability:.0%}):  {sm.bear.expected_return:.1%}")
        print(f"  Base ({sm.base.probability:.0%}):  {sm.base.expected_return:.1%}")
        print(f"  Bull ({sm.bull.probability:.0%}):  {sm.bull.expected_return:.1%}")
        print(f"  Expected Value:  {sm.expected_value:.1%}")
        print(f"\n-- INVESTMENT THESIS --")
        for line in v.thesis.split('\n'):
            print(f"  {line}")
        print(f"\n-- INVALIDATION CRITERIA --")
        for i, criterion in enumerate(v.invalidation_criteria, 1):
            print(f"  {i}. {criterion}")
        if result.portfolio_analysis:
            print(f"\n-- PORTFOLIO MANAGER --")
            print(f"  {result.portfolio_analysis.summary}")
            for finding in result.portfolio_analysis.key_findings[-3:]:
                print(f"  - {finding}")
        print(f"\n  Total runtime: {result.total_runtime_seconds:.1f}s")
        print(f"{sep}\n")
