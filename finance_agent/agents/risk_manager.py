"""
Agent 5: Risk Manager
Task: "What could go wrong, and how badly?"
Creates a structured Risk Register covering balance sheet, competitive,
valuation compression, macro/sector, and liquidity risks.
Quantifies downside using max drawdown, VaR, and bear case scenario.
"""

from __future__ import annotations

import json

from finance_agent.agents.base_agent import BaseAgent
from finance_agent.data.pipeline import AnalysisContext
from finance_agent.models.schemas import AgentAnalysis
from finance_agent.utils.logger import logger


SYSTEM_PROMPT = """You are a Chief Risk Officer and Portfolio Risk Manager at a systematic investment fund.

Your SOLE job is to identify, quantify, and rank risks. You are NOT a bull or a bear —
you are a dispassionate assessor of what could go wrong and how bad it would be.

Risk Register structure you always produce:
1. BALANCE SHEET RISK — leverage, liquidity, refinancing, covenant risk
2. COMPETITIVE RISK — moat erosion, new entrants, pricing pressure
3. VALUATION COMPRESSION RISK — multiple contraction in a risk-off environment
4. OPERATIONAL / EARNINGS RISK — margin compression, execution failure
5. MACRO / SECTOR RISK — interest rates, FX, regulation, cycle exposure
6. CONCENTRATION RISK — single product, customer, or geographic dependency

Scoring rubric (0-10, where HIGHER = SAFER):
10: Near-zero risk profile — fortress balance sheet, diversified, recession-resistant
8-9: Low risk — manageable risks, well-covered
6-7: Moderate risk — some notable risks but manageable
4-5: Elevated risk — material risks that require active monitoring
2-3: High risk — multiple risks could materialize simultaneously
0-1: Extreme risk — existential or near-term financial distress possible

Confidence: Your certainty in the risk assessment (0-1)."""

RESPONSE_SCHEMA = {
    "summary": "2-3 sentence executive summary of risk profile",
    "score": 6.5,
    "confidence": 0.80,
    "key_findings": [
        "Risk finding 1 with specific data",
        "Finding 2",
        "Finding 3",
        "Finding 4",
        "Finding 5"
    ],
    "risk_register": {
        "balance_sheet_risk": {
            "severity": "LOW/MEDIUM/HIGH",
            "assessment": "Specific assessment",
            "key_metric": "Net Debt/EBITDA: X"
        },
        "competitive_risk": {
            "severity": "LOW/MEDIUM/HIGH",
            "assessment": "Specific assessment",
            "key_metric": "Margin trend evidence"
        },
        "valuation_compression_risk": {
            "severity": "LOW/MEDIUM/HIGH",
            "assessment": "Specific assessment",
            "key_metric": "P/E at Xth percentile"
        },
        "earnings_risk": {
            "severity": "LOW/MEDIUM/HIGH",
            "assessment": "Specific assessment",
            "key_metric": "Earnings quality score"
        },
        "macro_risk": {
            "severity": "LOW/MEDIUM/HIGH",
            "assessment": "Specific assessment",
            "key_metric": "Beta + sector exposure"
        },
        "concentration_risk": {
            "severity": "LOW/MEDIUM/HIGH",
            "assessment": "Specific assessment",
            "key_metric": "Revenue diversification"
        }
    },
    "maximum_loss_estimate": "In a severe bear scenario, what is realistic downside? Support with data.",
    "altman_z_interpretation": "What does the Altman Z-Score tell us about distress probability?",
    "var_interpretation": "VaR interpretation and what it implies for position sizing",
    "top_risks_ranked": ["#1 Risk: ...", "#2 Risk: ...", "#3 Risk: ..."],
    "risk_mitigants": ["Factor that reduces risk 1", "Factor that reduces risk 2"]
}


class RiskManager(BaseAgent):
    name = "RiskManager"
    role_description = "Chief Risk Officer — structured risk register, downside quantification"

    def analyse(self, context: AnalysisContext) -> AgentAnalysis:
        logger.info(f"[{self.name}] Analysing {context.ticker}")

        metrics_block = self._format_metrics(context)
        d = context.to_prompt_dict()

        user_prompt = f"""Build a comprehensive Risk Register for {context.ticker} ({context.company_name}).

QUANTITATIVE DATA:
{metrics_block}

RISK-SPECIFIC DATA POINTS:
- Altman Z-Score: {d.get('altman_z_score', 'N/A')} (>2.99 = safe, 1.81-2.99 = grey, <1.81 = distress)
- Net Debt / EBITDA: {d.get('net_debt_to_ebitda', 'N/A')}
- Interest Coverage: {d.get('interest_coverage', 'N/A')}
- Current Ratio: {d.get('current_ratio', 'N/A')}
- Max Drawdown (1Y): {d.get('max_drawdown_1y', 'N/A')}
- VaR (95%, 1Y): {d.get('var_95_1y', 'N/A')}
- Beta: {d.get('beta', 'N/A')}
- Earnings Quality Score: {d.get('earnings_quality_score', 'N/A')} (0=low quality, 1=high quality)
- Piotroski F-Score: {d.get('piotroski_f_score', 'N/A')}/9
- P/E Percentile vs own history: {d.get('pe_percentile_5y', 'N/A')}th

BEAR CASE SCENARIO:
- Probability: {d.get('scenario_bear_prob', 'N/A')}
- Expected Return in Bear: {d.get('scenario_bear_return', 'N/A')}

Focus your analysis on:
1. Balance sheet sustainability — can the company weather a 12-18 month downturn?
2. Valuation risk — if multiples compress to historical lows, what is the downside?
3. Earnings quality risk — if cash conversion is low, is there an accruals risk?
4. Competitive moat risk — what evidence is there for or against margin erosion?
5. Macro sensitivity — how would rate rises, FX moves, or recession affect this company?
6. Scenario quantification — what are the realistic loss magnitudes?

{self._json_schema_instructions(RESPONSE_SCHEMA)}"""

        raw = self._call_llm_structured(SYSTEM_PROMPT, user_prompt)

        # Extract risk register summary
        rr = raw.get("risk_register", {})
        rr_findings = [
            f"Balance sheet risk: {rr.get('balance_sheet_risk', {}).get('severity', 'N/A')}",
            f"Competitive risk: {rr.get('competitive_risk', {}).get('severity', 'N/A')}",
            f"Valuation compression risk: {rr.get('valuation_compression_risk', {}).get('severity', 'N/A')}",
        ]

        return self._build_analysis(
            ticker=context.ticker,
            summary=raw.get("summary", "Risk analysis complete."),
            key_findings=raw.get("key_findings", []) + rr_findings + [
                f"Max loss estimate: {raw.get('maximum_loss_estimate', '')}",
            ],
            score=float(raw.get("score", 5.0)),
            confidence=float(raw.get("confidence", 0.5)),
            raw_output=json.dumps(raw, indent=2),
        )
