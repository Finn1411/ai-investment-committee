"""
Agent 3: Growth & Business Analyst
Task: "Can this business compound value over time?"
Evaluates: Revenue growth quality, pricing power, scalability, management
           execution, competitive advantages, unit economics, TAM.

Critical distinction: Narrative Growth vs. Fundamental Growth.
"""

from __future__ import annotations

import json

from finance_agent.agents.base_agent import BaseAgent
from finance_agent.data.pipeline import AnalysisContext
from finance_agent.models.schemas import AgentAnalysis
from finance_agent.utils.logger import logger


SYSTEM_PROMPT = """You are a senior Growth & Business Quality Analyst at a top investment fund.

Your job is to assess whether a company can compound value over the long term.
You focus on the QUALITY of growth, not just the headline numbers.

Key distinction you must always make:
- NARRATIVE GROWTH: Management promises, TAM stories, product launches without proof
- FUNDAMENTAL GROWTH: Revenue/FCF growth backed by improving unit economics, pricing power, market share

Always ask: "Is the growth rate sustainable, and is it value-creating?"
Growth that destroys capital (low ROIC growth) is worse than no growth.

Scoring rubric (0-10):
10: Exceptional compounder — durable high-quality growth, expanding TAM, pricing power, high ROIC reinvestment
8-9: Strong growth — above-average pace with good quality indicators
6-7: Solid — moderate growth, decent sustainability
4-5: Mixed — growth present but quality or sustainability questionable
2-3: Weak — slowing, low-quality, or capital-destructive growth
0-1: Declining or uninvestable growth trajectory

Confidence: Certainty in your growth assessment (0-1)."""

RESPONSE_SCHEMA = {
    "summary": "2-3 sentence executive summary of growth quality",
    "score": 6.0,
    "confidence": 0.75,
    "key_findings": [
        "Growth finding 1 with specific numbers",
        "Finding 2",
        "Finding 3",
        "Finding 4",
        "Finding 5"
    ],
    "growth_quality_assessment": "Is this narrative or fundamental growth? Explain.",
    "revenue_growth_sustainability": "Is the growth rate sustainable? What drives it?",
    "pricing_power_evidence": "Evidence for or against pricing power from margin trends",
    "scalability_assessment": "Can the business scale without proportional cost increases?",
    "management_execution": "What do the numbers suggest about management quality?",
    "competitive_advantage_assessment": "What is the source and durability of competitive edge?",
    "reinvestment_quality": "How effectively does management reinvest capital? (ROIC context)",
    "growth_risks": ["Risk 1", "Risk 2"],
    "growth_catalysts": ["Catalyst 1", "Catalyst 2"]
}


class GrowthAnalyst(BaseAgent):
    name = "GrowthAnalyst"
    role_description = "Senior Growth & Business Analyst — compounding quality, pricing power, competitive moat"

    def analyse(self, context: AnalysisContext) -> AgentAnalysis:
        logger.info(f"[{self.name}] Analysing {context.ticker}")

        metrics_block = self._format_metrics(context)
        d = context.to_prompt_dict()

        user_prompt = f"""Assess the growth quality and business compounding potential of {context.ticker} ({context.company_name}).

QUANTITATIVE DATA:
{metrics_block}

KEY GROWTH SIGNALS TO INTERPRET:
- Revenue Growth YoY: {d.get('revenue_growth_yoy', 'N/A')} | 3Y CAGR: {d.get('revenue_cagr_3y', 'N/A')}
- EPS Growth YoY: {d.get('eps_growth_yoy', 'N/A')} | 3Y CAGR: {d.get('eps_cagr_3y', 'N/A')}
- FCF Growth YoY: {d.get('fcf_growth_yoy', 'N/A')}
- Gross Margin Trend: {d.get('gross_margin_trend', 'N/A')} (positive = margin expansion)
- ROIC: {d.get('roic', 'N/A')} — the ultimate measure of reinvestment quality

Focus your analysis on:
1. Growth rate — is it accelerating or decelerating? What is driving it?
2. Quality test — is revenue growth translating into FCF growth? (If not, why not?)
3. Pricing power — do gross margins hold or expand even as the company grows?
4. ROIC interpretation — is management creating value with retained capital?
5. Moat sustainability — do the margin and ROIC levels suggest durable competitive advantage?
6. Management track record — what does the multi-year trend tell us?

Distinguish clearly between narrative claims and what the financial data actually proves.

{self._json_schema_instructions(RESPONSE_SCHEMA)}"""

        raw = self._call_llm_structured(SYSTEM_PROMPT, user_prompt)

        return self._build_analysis(
            ticker=context.ticker,
            summary=raw.get("summary", "Growth analysis complete."),
            key_findings=raw.get("key_findings", []) + [
                f"Growth quality: {raw.get('growth_quality_assessment', '')}",
                f"Competitive advantage: {raw.get('competitive_advantage_assessment', '')}",
            ],
            score=float(raw.get("score", 5.0)),
            confidence=float(raw.get("confidence", 0.5)),
            raw_output=json.dumps(raw, indent=2),
        )
