"""
Agent 2: Value Analyst
Task: "What expectations are already priced into the current stock price?"
Evaluates: Historical valuation ranges, FCF yield, DCF margin of safety,
           reverse-DCF implied growth, peer comparison context.
"""

from __future__ import annotations

import json

from finance_agent.agents.base_agent import BaseAgent
from finance_agent.data.pipeline import AnalysisContext
from finance_agent.models.schemas import AgentAnalysis
from finance_agent.utils.logger import logger


SYSTEM_PROMPT = """You are a senior Value Analyst specializing in valuation and return expectations.

Your central question is ALWAYS: "What does the current stock price imply about future growth,
and is that expectation reasonable or mispriced?"

You do NOT forecast stock prices. You assess whether the current price offers:
- An adequate margin of safety
- A reasonable risk/reward setup
- Rational embedded growth expectations

Scoring rubric (0-10):
10: Deeply undervalued — significant margin of safety, market pricing in pessimism
8-9: Attractively valued — below intrinsic value with good risk/reward
6-7: Fairly valued — priced for reasonable outcomes, limited upside
4-5: Richly valued — premium price requiring aggressive assumptions to justify
2-3: Expensive — market pricing in optimistic growth that may not materialize
0-1: Dangerously overvalued — requires heroic assumptions to justify current price

Confidence: Your certainty in the valuation assessment (0-1)."""

RESPONSE_SCHEMA = {
    "summary": "2-3 sentence executive summary of valuation",
    "score": 4.0,
    "confidence": 0.80,
    "key_findings": [
        "Valuation finding 1 with specific numbers",
        "Finding 2",
        "Finding 3",
        "Finding 4",
        "Finding 5"
    ],
    "embedded_expectations": "What growth rate / ROIC does the current price imply?",
    "margin_of_safety_assessment": "Is there a margin of safety? Explain.",
    "fcf_yield_assessment": "Is the FCF yield attractive vs alternatives?",
    "historical_valuation_context": "How does current valuation compare to company's own history?",
    "reverse_dcf_interpretation": "What does the reverse-DCF tell us? Is the implied growth achievable?",
    "valuation_risks": ["Risk 1", "Risk 2"],
    "valuation_opportunities": ["Opportunity 1 or None"],
    "base_case_return_expectation": "Analyst's qualitative view on expected return"
}


class ValueAnalyst(BaseAgent):
    name = "ValueAnalyst"
    role_description = "Senior Value Analyst — pricing, margin of safety, embedded expectations"

    def analyse(self, context: AnalysisContext) -> AgentAnalysis:
        logger.info(f"[{self.name}] Analysing {context.ticker}")

        metrics_block = self._format_metrics(context)
        d = context.to_prompt_dict()

        sensitivity_str = ""
        if context.valuation and context.valuation.sensitivity:
            try:
                df = context.valuation.sensitivity.to_dataframe()
                sensitivity_str = f"\nDCF SENSITIVITY TABLE (intrinsic value per share):\n{df.to_string()}"
            except Exception:
                pass

        user_prompt = f"""Perform a deep valuation analysis of {context.ticker} ({context.company_name}).

QUANTITATIVE DATA:
{metrics_block}
{sensitivity_str}

CURRENT VALUATION LABEL (from DCF model): {d.get('valuation_label', 'N/A')}
DCF INTRINSIC VALUE: ${d.get('dcf_intrinsic_per_share', 'N/A')}
MARGIN OF SAFETY: {d.get('dcf_margin_of_safety', 'N/A')}
REVERSE-DCF: {d.get('reverse_dcf_narrative', 'N/A')}

Focus your analysis on:
1. Is the stock cheap or expensive relative to intrinsic value?
2. What growth rate is the market pricing in (from reverse-DCF)? Is it realistic?
3. Where does the current valuation sit vs. the company's own 2-year history?
4. What FCF yield does the investor receive? How does it compare to risk-free rate?
5. What is the risk/reward asymmetry? (Downside in bear case vs upside in bull case)
6. Sensitivity analysis: which scenario would make this attractive?

{self._json_schema_instructions(RESPONSE_SCHEMA)}"""

        raw = self._call_llm_structured(SYSTEM_PROMPT, user_prompt)

        return self._build_analysis(
            ticker=context.ticker,
            summary=raw.get("summary", "Valuation analysis complete."),
            key_findings=raw.get("key_findings", []) + [
                f"Embedded expectations: {raw.get('embedded_expectations', '')}",
                f"Return expectation: {raw.get('base_case_return_expectation', '')}",
            ],
            score=float(raw.get("score", 5.0)),
            confidence=float(raw.get("confidence", 0.5)),
            raw_output=json.dumps(raw, indent=2),
        )
