"""
Agent 4: Earnings & Catalyst Analyst
Task: "What near-term events could move this stock significantly?"
Evaluates: Upcoming earnings, guidance changes, margin expansion/compression,
           buyback activity, M&A potential, insider activity, product launches.
"""

from __future__ import annotations

import json

from finance_agent.agents.base_agent import BaseAgent
from finance_agent.data.pipeline import AnalysisContext
from finance_agent.models.schemas import AgentAnalysis
from finance_agent.utils.logger import logger


SYSTEM_PROMPT = """You are a senior Earnings & Catalyst Analyst specializing in identifying
near-term event-driven opportunities and risks.

You focus on EVENTS and CHANGES, not static quality.

Your key questions:
1. What could change the market's view of this stock in the next 6-12 months?
2. Is the company set up to beat or disappoint earnings expectations?
3. Are there structural catalysts (buybacks, M&A, margin inflection) not yet priced in?
4. Are there risks of negative surprises that the bull case ignores?

Scoring rubric (0-10):
10: Strong upcoming catalysts with high probability, minimal near-term risk
8-9: Clear positive catalysts with manageable risk profile
6-7: Mixed — some catalysts but uncertainty is high
4-5: Limited near-term catalysts, significant risk of disappointment
2-3: Headwinds dominate, elevated disappointment risk
0-1: Multiple negative catalysts converging, near-term very risky

Confidence: Your certainty about the catalyst picture (0-1)."""

RESPONSE_SCHEMA = {
    "summary": "2-3 sentence summary of the catalyst picture",
    "score": 6.0,
    "confidence": 0.65,
    "key_findings": [
        "Catalyst finding 1",
        "Finding 2",
        "Finding 3",
        "Finding 4",
        "Finding 5"
    ],
    "earnings_setup": "Is the company set up to beat or disappoint? Based on what evidence?",
    "margin_catalyst": "Is there a margin expansion/compression catalyst? Evidence?",
    "capital_return_catalyst": "Buybacks, dividends — are they value-accretive at current prices?",
    "structural_catalysts": ["Positive structural catalyst 1", "Positive structural catalyst 2"],
    "near_term_risks": ["Near-term risk 1", "Near-term risk 2"],
    "guidance_watch": "What should investors watch for in next earnings guidance?",
    "time_horizon_assessment": "Which catalyst is most likely to materialize and when?"
}


class EarningsAnalyst(BaseAgent):
    name = "EarningsAnalyst"
    role_description = "Senior Earnings & Catalyst Analyst — event-driven opportunities and risks"

    def analyse(self, context: AnalysisContext) -> AgentAnalysis:
        logger.info(f"[{self.name}] Analysing {context.ticker}")

        metrics_block = self._format_metrics(context)
        d = context.to_prompt_dict()

        # Format earnings dates if available
        earnings_info = "No upcoming earnings date available."
        if context.raw.earnings_dates is not None and not context.raw.earnings_dates.empty:
            try:
                recent = context.raw.earnings_dates.head(3)
                earnings_info = f"Recent/upcoming earnings dates:\n{recent.to_string()}"
            except Exception:
                pass

        user_prompt = f"""Analyse the catalyst picture and earnings setup for {context.ticker} ({context.company_name}).

QUANTITATIVE DATA:
{metrics_block}

EARNINGS CALENDAR:
{earnings_info}

SCENARIO CONTEXT:
- Market's expected return (EV): {d.get('scenario_expected_value', 'N/A')}
- Bull probability: {d.get('scenario_bull_prob', 'N/A')} | Bull return: {d.get('scenario_bull_return', 'N/A')}
- Bear probability: {d.get('scenario_bear_prob', 'N/A')} | Bear return: {d.get('scenario_bear_return', 'N/A')}

Focus your analysis on:
1. Earnings setup: Does FCF growth diverging from EPS signal anything? Is earnings quality high?
2. Margin trajectory: Is operating leverage improving? What does gross margin trend signal?
3. Capital allocation: Are buybacks happening at expensive multiples? Is FCF yield enabling returns?
4. Structural catalysts: What product cycles, market expansions, or cost initiatives could drive upside?
5. Disappointment risks: Where could the company fall short of consensus expectations?
6. FCF growth vs. EPS growth divergence — what story does it tell?

{self._json_schema_instructions(RESPONSE_SCHEMA)}"""

        raw = self._call_llm_structured(SYSTEM_PROMPT, user_prompt)

        return self._build_analysis(
            ticker=context.ticker,
            summary=raw.get("summary", "Catalyst analysis complete."),
            key_findings=raw.get("key_findings", []) + [
                f"Earnings setup: {raw.get('earnings_setup', '')}",
                f"Key catalyst: {raw.get('time_horizon_assessment', '')}",
            ],
            score=float(raw.get("score", 5.0)),
            confidence=float(raw.get("confidence", 0.5)),
            raw_output=json.dumps(raw, indent=2),
        )
