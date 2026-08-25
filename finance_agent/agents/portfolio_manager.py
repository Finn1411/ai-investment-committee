"""
Agent 7: Portfolio Manager
Task: "Does this stock make sense in the context of my portfolio?"

An excellent stock can still be a bad portfolio decision.
The Portfolio Manager considers: position sizing, correlation, sector weights,
concentration risk, risk-adjusted return contribution, and fit with existing holdings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from finance_agent.agents.base_agent import BaseAgent
from finance_agent.data.pipeline import AnalysisContext
from finance_agent.models.schemas import AgentAnalysis
from finance_agent.utils.logger import logger


@dataclass
class PortfolioContext:
    """
    Represents the current portfolio state.
    Passed to PortfolioManager to evaluate fit.
    """
    existing_positions: dict[str, float] = field(default_factory=dict)    # ticker -> weight %
    sector_weights: dict[str, float] = field(default_factory=dict)        # sector -> weight %
    total_value_usd: Optional[float] = None
    target_sector_max: float = 0.30     # Max 30% in any sector
    target_position_max: float = 0.10   # Max 10% single position
    target_position_min: float = 0.02   # Min 2% to be meaningful
    benchmark: str = "S&P 500"
    risk_budget_used_pct: float = 0.0   # 0-100: how much risk budget is already deployed


SYSTEM_PROMPT = """You are a Portfolio Manager responsible for the overall portfolio construction.

Your job is NOT to decide if a stock is good — other analysts did that.
Your job is: "Does adding this stock to my portfolio improve the portfolio?"

You evaluate:
1. POSITION SIZING: What % weight makes sense given risk and conviction?
2. CORRELATION: Does this stock add diversification or increase concentration?
3. SECTOR WEIGHT: Does adding it push sector weight above limits?
4. RISK CONTRIBUTION: Does it increase or decrease portfolio risk meaningfully?
5. ALTERNATIVE COST: What would you sell to fund this? Is the trade-off worth it?
6. PORTFOLIO FIT SCORE: Does it complement or duplicate existing holdings?

The Portfolio Manager can VETO a BUY recommendation if:
- Position would breach sector concentration limits
- Risk/reward is dominated by existing holdings
- Portfolio already has too many similar high-beta/growth positions

Scoring rubric (0-10):
10: Perfect fit — diversifies, appropriate sizing, high conviction
8-9: Strong fit — adds value to portfolio with minor concentration trade-off
6-7: Decent fit — works but some concentration or overlap concern
4-5: Neutral — stock may be good but portfolio fit is questionable
2-3: Poor fit — would create problematic concentration or correlation
0-1: Veto — portfolio constraints prohibit position

Confidence: Your certainty in the portfolio fit assessment (0-1)."""

RESPONSE_SCHEMA = {
    "summary": "2-3 sentence portfolio fit assessment",
    "score": 7.0,
    "confidence": 0.75,
    "key_findings": [
        "Portfolio finding 1",
        "Finding 2",
        "Finding 3",
        "Finding 4",
        "Finding 5"
    ],
    "recommended_position_size_pct": 4.0,
    "position_sizing_rationale": "Why this specific size given risk/conviction/constraints",
    "sector_concentration_assessment": "Does this create problematic sector concentration?",
    "correlation_assessment": "Is this correlated with existing holdings? Diversifying?",
    "risk_contribution_assessment": "How does this change portfolio risk profile?",
    "portfolio_fit_verdict": "BUY / ACCUMULATE / HOLD / TRIM / AVOID",
    "conditions_for_larger_position": "What would justify a larger position?",
    "conditions_for_no_position": "What would make you avoid this entirely?",
    "alternative_opportunity_cost": "What is the opportunity cost of choosing this vs alternatives?"
}


class PortfolioManager(BaseAgent):
    name = "PortfolioManager"
    role_description = "Portfolio Manager — position sizing, correlation, portfolio fit"

    def analyse(
        self,
        context: AnalysisContext,
        portfolio: Optional[PortfolioContext] = None,
        committee_score: Optional[float] = None,
    ) -> AgentAnalysis:
        """
        Args:
            context: Full AnalysisContext
            portfolio: Current portfolio state (uses default if None)
            committee_score: Weighted score from the committee (0-10)
        """
        logger.info(f"[{self.name}] Analysing {context.ticker} portfolio fit")

        portfolio = portfolio or PortfolioContext()
        metrics_block = self._format_metrics(context)
        d = context.to_prompt_dict()

        # Format portfolio state
        positions_str = "\n".join(
            [f"  {t}: {w:.1%}" for t, w in portfolio.existing_positions.items()]
        ) or "  No current positions"

        sector_weights_str = "\n".join(
            [f"  {s}: {w:.1%}" for s, w in portfolio.sector_weights.items()]
        ) or "  No sector data"

        user_prompt = f"""Evaluate the portfolio fit for adding {context.ticker} ({context.company_name}).

STOCK DATA:
{metrics_block}

PORTFOLIO CONSTRAINTS:
- Max single position: {portfolio.target_position_max:.0%}
- Min meaningful position: {portfolio.target_position_min:.0%}
- Max sector weight: {portfolio.target_sector_max:.0%}
- Risk budget used: {portfolio.risk_budget_used_pct:.0f}%
- Benchmark: {portfolio.benchmark}

CURRENT PORTFOLIO POSITIONS:
{positions_str}

CURRENT SECTOR WEIGHTS:
{sector_weights_str}

THIS STOCK:
- Sector: {context.sector} | Industry: {context.industry}
- Beta: {d.get('beta', 'N/A')} | Volatility: {d.get('volatility_annual', 'N/A')}
- Expected Value (EV): {d.get('scenario_expected_value', 'N/A')}
- Committee Score: {committee_score}/10 if committee_score else "Not yet scored"
- Valuation: {d.get('valuation_label', 'N/A')}
- FCF Yield: {d.get('fcf_yield', 'N/A')} (income component)
- Max Drawdown 1Y: {d.get('max_drawdown_1y', 'N/A')} (tail risk)

Focus your analysis on:
1. Optimal position size given risk/reward and conviction from committee score
2. Sector concentration — does {context.sector} sector need more/less exposure?
3. Risk contribution — does this stock add systematic or idiosyncratic risk?
4. Correlation with existing holdings — does it diversify or concentrate?
5. Alternative cost — is there a better use of portfolio capital?
6. Conditions for size changes — when would you add more or cut?

{self._json_schema_instructions(RESPONSE_SCHEMA)}"""

        raw = self._call_llm_structured(SYSTEM_PROMPT, user_prompt)

        return self._build_analysis(
            ticker=context.ticker,
            summary=raw.get("summary", "Portfolio fit analysis complete."),
            key_findings=raw.get("key_findings", []) + [
                f"Recommended size: {raw.get('recommended_position_size_pct', 'N/A')}%",
                f"Portfolio verdict: {raw.get('portfolio_fit_verdict', 'N/A')}",
                f"Rationale: {raw.get('position_sizing_rationale', '')}",
            ],
            score=float(raw.get("score", 5.0)),
            confidence=float(raw.get("confidence", 0.5)),
            raw_output=json.dumps(raw, indent=2),
        )
