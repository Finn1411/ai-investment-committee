"""
Agent 1: Fundamental Analyst
Task: "How healthy and economically strong is the company?"
Evaluates: Business quality, financial strength, profitability trends,
           competitive position, core strengths and weaknesses.
"""

from __future__ import annotations

import json

from finance_agent.agents.base_agent import BaseAgent
from finance_agent.data.pipeline import AnalysisContext
from finance_agent.models.schemas import AgentAnalysis
from finance_agent.utils.logger import logger


SYSTEM_PROMPT = """You are a senior Fundamental Analyst at a top-tier investment firm.
Your job is to assess the intrinsic business quality and financial health of a company.

You operate strictly within the data provided — you do NOT speculate beyond the metrics.
Your analysis must be evidence-based, citing specific numbers from the data.

Scoring rubric (0-10):
10: Exceptional business — dominant moat, expanding margins, fortress balance sheet, high ROIC
8-9: Strong business — above-average quality with minor concerns
6-7: Solid business — decent fundamentals with notable weaknesses
4-5: Mixed — some strengths offset by material concerns
2-3: Weak — significant financial or competitive problems
0-1: Distressed or uninvestable

Confidence is your certainty in the score (0-1) based on data completeness and consistency."""

RESPONSE_SCHEMA = {
    "summary": "2-3 sentence executive summary of fundamental health",
    "score": 7.5,
    "confidence": 0.85,
    "key_findings": [
        "Finding 1 with specific numbers",
        "Finding 2 with specific numbers",
        "Finding 3 with specific numbers",
        "Finding 4 with specific numbers",
        "Finding 5 with specific numbers"
    ],
    "sources": ["List of sources used from the RAG claims or quantitative data"],
    "strengths": ["Strength 1", "Strength 2", "Strength 3"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "business_quality_assessment": "one paragraph assessment",
    "financial_health_assessment": "one paragraph assessment",
    "trend_assessment": "Are margins/returns improving or deteriorating? Why?",
    "moat_indicators": "Evidence for or against durable competitive advantage",
    "red_flags": ["Any red flag or None"]
}


class FundamentalAnalyst(BaseAgent):
    name = "FundamentalAnalyst"
    role_description = "Senior Fundamental Analyst — business quality and financial health"

    def analyse(self, context: AnalysisContext) -> AgentAnalysis:
        logger.info(f"[{self.name}] Analysing {context.ticker}")

        metrics_block = self._format_metrics(context)
        user_prompt = f"""Analyse the fundamental quality of {context.ticker} ({context.company_name}).

QUANTITATIVE DATA (all calculations are pre-verified — trust these numbers):
{metrics_block}

Data Quality: {'PASSED' if context.data_passed_quality_check else 'HAS WARNINGS'}
Warnings: {context.quality_report.warnings if context.quality_report.warnings else 'None'}

Focus your analysis on:
1. Profitability: Are margins high and improving? How does ROIC compare to cost of capital?
2. Cash Flow Quality: Is earnings quality high? Does FCF conversion support the business?
3. Balance Sheet: Is the company over-leveraged? Can it survive a downturn?
4. Competitive Position: What do the margin levels and ROIC imply about competitive advantage?
5. Trend Analysis: Are fundamentals improving or deteriorating? What is the direction of travel?
6. Piotroski (financial strength) and Altman Z (bankruptcy risk) interpretation.

{self._json_schema_instructions(RESPONSE_SCHEMA)}"""

        raw = self._call_llm_structured(SYSTEM_PROMPT, user_prompt)

        return self._build_analysis(
            ticker=context.ticker,
            summary=raw.get("summary", "Fundamental analysis complete."),
            key_findings=raw.get("key_findings", []) + [
                f"Business quality: {raw.get('business_quality_assessment', '')}",
                f"Moat indicators: {raw.get('moat_indicators', '')}",
            ],
            score=float(raw.get("score", 5.0)),
            confidence=float(raw.get("confidence", 0.5)),
            raw_output=json.dumps(raw, indent=2),
            sources=raw.get("sources", []),
        )
