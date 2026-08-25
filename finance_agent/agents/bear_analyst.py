"""
Agent 6: Bear Analyst / Devil's Advocate
Task: "Why could this investment thesis be WRONG?"

This agent is the most important safeguard against groupthink.
It receives the COMBINED positive analyses and must actively try to REFUTE them.
It must not summarize what other agents said — it must challenge their conclusions.

Inspired by: Devil's advocate processes at Bridgewater, Tiger, etc.
"""

from __future__ import annotations

import json

from finance_agent.agents.base_agent import BaseAgent
from finance_agent.data.pipeline import AnalysisContext
from finance_agent.models.schemas import AgentAnalysis
from finance_agent.utils.logger import logger


SYSTEM_PROMPT = """You are the Devil's Advocate at an investment committee.
Your SOLE purpose is to challenge the bull thesis.

RULES:
1. You MUST actively try to refute the investment case — not just list generic risks
2. You MUST cite specific numbers that the bulls might be ignoring or misinterpreting
3. You MUST challenge the assumptions embedded in the DCF / growth projections
4. You MUST consider what needs to go RIGHT for the bull case, and ask: "Is that realistic?"
5. You must NOT simply list the same risks as the Risk Manager — you must find new angles
6. You are NOT trying to be balanced — you are trying to find holes in the thesis

Your most powerful tools:
- "The market is pricing in X% growth. What if it's Y%?"
- "Margins look strong, but what happens when [competitor/macro/regulation] hits?"
- "The Piotroski score is high, but signal Z is actually concerning because..."
- "Revenue is growing but FCF is not — this is a red flag because..."
- "The valuation percentile is high — this is not just risk, it mathematically limits upside"

Scoring rubric (0-10, where LOWER = MORE BEARISH, i.e., thesis is WEAKER):
0-2: Thesis has fatal flaws — the bear case is compelling and likely
3-4: Bear case has strong specific evidence — requires significant risk premium
5-6: Mixed — bear case has some merit but is not dominant
7-8: Bear case is relatively weak — thesis holds under stress testing
9-10: Bear case is very difficult to make — thesis is highly robust

Confidence: Your certainty in the bear case (0-1)."""

RESPONSE_SCHEMA = {
    "summary": "2-3 sentence bear thesis — what is the core bear argument?",
    "score": 4.0,
    "confidence": 0.70,
    "key_findings": [
        "Bear finding 1 — specific refutation with numbers",
        "Finding 2",
        "Finding 3",
        "Finding 4",
        "Finding 5"
    ],
    "thesis_vulnerabilities": [
        "Specific vulnerability in the bull thesis — with data",
        "Vulnerability 2",
        "Vulnerability 3"
    ],
    "dcf_assumptions_challenged": "What assumptions in the DCF are too optimistic? Why?",
    "growth_story_challenged": "What specific growth assumptions could fail?",
    "what_must_go_right": "List the key assumptions that MUST hold for the bull case",
    "historical_analog": "Is there a historical parallel of a company priced like this that disappointed?",
    "ignored_red_flags": ["Red flag the bulls are dismissing 1", "Red flag 2"],
    "probability_weighted_bear_case": "If bear thesis materializes, what is the realistic downside?",
    "what_would_make_bear_wrong": "What would disprove the bear case? What data to watch?"
}


class BearAnalyst(BaseAgent):
    name = "BearAnalyst"
    role_description = "Devil's Advocate — challenges bull thesis, finds thesis vulnerabilities"

    def analyse(
        self,
        context: AnalysisContext,
        bull_thesis_summary: str = "",
    ) -> AgentAnalysis:
        """
        Args:
            context: Full AnalysisContext
            bull_thesis_summary: Optional combined summary from other agents to refute
        """
        logger.info(f"[{self.name}] Analysing {context.ticker} (challenging bull thesis)")

        metrics_block = self._format_metrics(context)
        d = context.to_prompt_dict()

        bull_context = ""
        if bull_thesis_summary:
            bull_context = f"""
THE BULL THESIS TO REFUTE:
{bull_thesis_summary}

Your job: Find the holes in this thesis. What are the bulls missing?
"""

        user_prompt = f"""Challenge the investment thesis for {context.ticker} ({context.company_name}).
{bull_context}

QUANTITATIVE DATA (find what the bulls are ignoring or misinterpreting):
{metrics_block}

KEY NUMBERS TO STRESS TEST:
- DCF intrinsic value: ${d.get('dcf_intrinsic_per_share', 'N/A')} (what if discount rate is 12% not 10%?)
- Implied FCF growth from reverse-DCF: {d.get('implied_fcf_growth', 'N/A')} — is this achievable?
- P/E at {d.get('pe_percentile_5y', 'N/A')}th historical percentile — what does compression mean?
- FCF growth YoY: {d.get('fcf_growth_yoy', 'N/A')} — diverging from EPS growth?
- Cash conversion: {d.get('cash_conversion', 'N/A')} — are earnings real?
- Gross margin trend: {d.get('gross_margin_trend', 'N/A')} — compressing or expanding?
- Beta {d.get('beta', 'N/A')} — what happens in a market correction at this valuation?
- Bear case return: {d.get('scenario_bear_return', 'N/A')} with {d.get('scenario_bear_prob', 'N/A')} probability

Your mandate:
1. Find the most dangerous assumption in the bull thesis
2. Use specific numbers to show where the math breaks down
3. Identify what is NOT in the data that should concern us
4. Challenge the growth sustainability with FCF/EPS divergence analysis
5. Estimate realistic downside if your bear case materializes
6. What would change your mind?

{self._json_schema_instructions(RESPONSE_SCHEMA)}"""

        raw = self._call_llm_structured(SYSTEM_PROMPT, user_prompt)

        return self._build_analysis(
            ticker=context.ticker,
            summary=raw.get("summary", "Bear case analysis complete."),
            key_findings=raw.get("key_findings", []) + [
                f"Key vulnerability: {raw.get('thesis_vulnerabilities', [''])[0] if raw.get('thesis_vulnerabilities') else ''}",
                f"What must go right: {raw.get('what_must_go_right', '')}",
                f"Downside: {raw.get('probability_weighted_bear_case', '')}",
            ],
            score=float(raw.get("score", 5.0)),
            confidence=float(raw.get("confidence", 0.5)),
            raw_output=json.dumps(raw, indent=2),
        )
