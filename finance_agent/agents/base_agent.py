"""
Abstract base class for all Finance Agents.
Enforces a consistent interface: analyse(context) -> AgentAnalysis.

Key design principles:
- Agents receive AnalysisContext (pre-validated, all metrics computed)
- LLMs only interpret — never calculate
- All LLM responses are parsed into structured JSON
- Graceful fallback if JSON parsing fails
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Optional

import google.genai as genai
from google.genai import types as genai_types

from finance_agent.data.pipeline import AnalysisContext
from finance_agent.models.schemas import AgentAnalysis
from finance_agent.utils.config import settings
from finance_agent.utils.logger import logger


class BaseAgent(ABC):
    """
    All research agents inherit from this class.

    Provides:
    - Shared Gemini client with structured output
    - _call_llm_structured(): calls LLM, extracts JSON response
    - _build_analysis(): constructs validated AgentAnalysis
    - analyse(): abstract contract all agents must implement
    """

    name: str = "BaseAgent"
    # Each agent sets its own system role description
    role_description: str = "You are a professional financial analyst."

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning(f"[{self.name}] GEMINI_API_KEY not set -- LLM calls will fail.")
        self._client = genai.Client(api_key=api_key)
        self._model_cascade = list(settings.llm.model_cascade)
        self._model_name = self._model_cascade[0]
        self._gen_config = genai_types.GenerateContentConfig(
            temperature=settings.llm.temperature,
            max_output_tokens=settings.llm.max_tokens,
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        logger.debug(f"[{self.name}] Initialised with model {self._model_name}")

    def _advance_model(self) -> bool:
        """Try the next model in the cascade. Returns True if a fallback is available."""
        current_idx = self._model_cascade.index(self._model_name)
        if current_idx + 1 < len(self._model_cascade):
            self._model_name = self._model_cascade[current_idx + 1]
            logger.warning(f"[{self.name}] Falling back to model: {self._model_name}")
            return True
        return False

    @abstractmethod
    def analyse(self, context: AnalysisContext) -> AgentAnalysis:
        """
        Run the agent's full analysis.

        Args:
            context: AnalysisContext from DataPipeline.run()

        Returns:
            AgentAnalysis with score, confidence, findings, summary
        """
        ...

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _call_llm_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        expected_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Call Gemini and parse the response as structured JSON.

        The prompt instructs the LLM to respond with a JSON object.
        We extract and validate the JSON from the response.

        Returns:
            Parsed dict, or {"raw": response_text} on parse failure.
        """
        import time
        full_prompt = (
            f"{system_prompt}\n\n"
            "CRITICAL: Your entire response MUST be valid JSON. "
            "Do not include any text before or after the JSON. "
            "Do not use markdown code fences.\n\n"
            f"---\n\n{user_prompt}"
        )
        max_retries = 3
        retry_delays = [5, 15, 30]
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=full_prompt,
                    config=self._gen_config,
                )
                text = response.text.strip()

                # Strip markdown code fences if present
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

                parsed = json.loads(text)
                logger.debug(f"[{self.name}] JSON response parsed OK")
                return parsed

            except json.JSONDecodeError as e:
                logger.warning(f"[{self.name}] JSON parse failed: {e} -- using raw fallback")
                try:
                    match = re.search(r"\{.*\}", text, re.DOTALL)
                    if match:
                        return json.loads(match.group())
                except Exception:
                    pass
                return {"raw": text, "parse_error": str(e)}

            except Exception as exc:
                error_str = str(exc)
                is_daily_quota = "PerDay" in error_str or "per_day" in error_str.lower() or "Free Tier" in error_str
                is_model_unavailable = "404" in error_str or is_daily_quota
                is_retryable = ("503" in error_str or "UNAVAILABLE" in error_str or
                                ("429" in error_str and not is_daily_quota))
                # Try next model in cascade for 404s / daily quota exhaustion
                if is_model_unavailable and self._advance_model():
                    logger.warning(f"[{self.name}] Retrying with fallback model {self._model_name}")
                    continue  # retry immediately with new model
                if is_retryable and attempt < max_retries - 1:
                    wait = retry_delays[attempt]
                    logger.warning(
                        f"[{self.name}] Transient error (attempt {attempt+1}/{max_retries}), "
                        f"retrying in {wait}s... [{error_str[:80]}]"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"[{self.name}] LLM call failed: {exc}")
                raise

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Simple LLM call returning raw text (for narrative sections)."""
        import time
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        max_retries = 3
        retry_delays = [5, 15, 30]
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=full_prompt,
                    config=self._gen_config,
                )
                return response.text
            except Exception as exc:
                error_str = str(exc)
                is_daily_quota = "PerDay" in error_str or "per_day" in error_str.lower() or "Free Tier" in error_str
                is_model_unavailable = "404" in error_str or is_daily_quota
                is_retryable = ("503" in error_str or "UNAVAILABLE" in error_str or
                                ("429" in error_str and not is_daily_quota))
                if is_model_unavailable and self._advance_model():
                    logger.warning(f"[{self.name}] Retrying with fallback model {self._model_name}")
                    continue
                if is_retryable and attempt < max_retries - 1:
                    wait = retry_delays[attempt]
                    logger.warning(
                        f"[{self.name}] Transient error (attempt {attempt+1}/{max_retries}), "
                        f"retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"[{self.name}] LLM call failed: {exc}")
                raise

    def _build_analysis(
        self,
        ticker: str,
        summary: str,
        key_findings: list[str],
        score: float,
        confidence: float,
        raw_output: str = "",
        sources: list[str] | None = None,
    ) -> AgentAnalysis:
        """Construct a validated AgentAnalysis."""
        return AgentAnalysis(
            agent_name=self.name,
            ticker=ticker,
            analysis_date=date.today(),
            summary=summary,
            key_findings=key_findings,
            score=round(max(0.0, min(10.0, score)), 2),
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            raw_output=raw_output,
            sources=sources or [],
        )

    # ── Prompt utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _format_metrics(ctx: AnalysisContext) -> str:
        """Format key metrics into a clean LLM-readable block."""
        d = ctx.to_prompt_dict()

        def pct(v): return f"{v:.1%}" if v is not None else "N/A"
        def num(v, dec=2): return f"{v:.{dec}f}" if v is not None else "N/A"
        def bn(v): return f"${v:.1f}B" if v is not None else "N/A"

        lines = [
            f"COMPANY: {ctx.company_name} | SECTOR: {ctx.sector} | INDUSTRY: {ctx.industry}",
            f"PRICE: ${num(d.get('current_price'))} | MARKET CAP: {bn(d.get('market_cap_bn'))}",
            "",
            "-- MARKET METRICS --",
            f"Return 1Y: {pct(d.get('return_1y'))} | YTD: {pct(d.get('return_ytd'))}",
            f"Beta: {num(d.get('beta'))} | Volatility (90d): {pct(d.get('volatility_annual'))}",
            f"Sharpe 1Y: {num(d.get('sharpe_ratio_1y'))} | Max Drawdown 1Y: {pct(d.get('max_drawdown_1y'))}",
            "",
            "-- PROFITABILITY --",
            f"Gross Margin: {pct(d.get('gross_margin'))} | Op Margin: {pct(d.get('operating_margin'))} | Net Margin: {pct(d.get('net_margin'))}",
            f"ROIC: {pct(d.get('roic'))} | ROE: {pct(d.get('roe'))}",
            f"FCF Margin: {pct(d.get('fcf_margin'))} | FCF Yield: {pct(d.get('fcf_yield'))}",
            f"Cash Conversion: {num(d.get('cash_conversion'))} | Earnings Quality: {num(d.get('earnings_quality_score'))}",
            "",
            "-- BALANCE SHEET --",
            f"Net Debt/EBITDA: {num(d.get('net_debt_to_ebitda'))} | Interest Coverage: {num(d.get('interest_coverage'))} | Current Ratio: {num(d.get('current_ratio'))}",
            "",
            "-- GROWTH --",
            f"Revenue YoY: {pct(d.get('revenue_growth_yoy'))} | EPS YoY: {pct(d.get('eps_growth_yoy'))} | FCF YoY: {pct(d.get('fcf_growth_yoy'))}",
            f"Revenue CAGR 3Y: {pct(d.get('revenue_cagr_3y'))} | EPS CAGR 3Y: {pct(d.get('eps_cagr_3y'))}",
            f"Gross Margin Trend: {num(d.get('gross_margin_trend'), 3)} (positive = improving)",
            "",
            "-- VALUATION --",
            f"P/E: {num(d.get('pe_ratio'))} | Fwd P/E: {num(d.get('forward_pe'))} | EV/EBITDA: {num(d.get('ev_to_ebitda'))} | P/S: {num(d.get('price_to_sales'))}",
            f"P/E Percentile (vs 2Y history): {num(d.get('pe_percentile_5y'), 0)}th",
            f"FCF Yield: {pct(d.get('fcf_yield'))}",
            f"Intrinsic DCF: ${num(d.get('dcf_intrinsic_per_share'))} | Implied FCF Growth: {pct(d.get('implied_fcf_growth'))}",
            "",
            "-- QUALITY & WARNINGS --",
            f"Overall Quality: {num(d.get('overall_quality_score'))}/10",
        ]

        if ctx.quality_report.warnings:
            lines.append("Warnings: " + ", ".join(ctx.quality_report.warnings[:3]))

        # -- ADD MACRO REGIME & PEER CONTEXT (Phase 9) --
        if ctx.macro_regime_summary:
            lines.append("")
            lines.append("-- MACRO ENVIRONMENT --")
            lines.append(ctx.macro_regime_summary)
            
        if ctx.peer_context:
            lines.append("")
            lines.append("-- SECTOR / PEER BENCHMARKING --")
            lines.append(ctx.peer_context)

        # -- ADD RAG CLAIMS (Phase 8) --
        if ctx.claims:
            lines.append("")
            lines.append("-- RECENT NEWS & FACTUAL CLAIMS (RAG) --")
            for c in ctx.claims:
                meta = c.get('metadata', {})
                lines.append(f"• {c.get('claim')} (Source: {meta.get('source_title', 'Unknown')} | Tier: {meta.get('tier', 'Unknown')})")

        if d.get("dcf_intrinsic_per_share"):
            lines += [
                "",
                "-- DCF MODEL --",
                f"Intrinsic Value: ${num(d.get('dcf_intrinsic_per_share'))} | Margin of Safety: {pct(d.get('dcf_margin_of_safety'))}",
                f"Valuation Label: {d.get('valuation_label')} | {d.get('reverse_dcf_narrative', '')}",
            ]

        lines += [
            "",
            "-- COMPOSITE SCORES --",
            f"Piotroski F-Score: {d.get('piotroski_f_score')}/9 | Altman Z: {num(d.get('altman_z_score'))} | Quality: {num(d.get('overall_quality_score'))}/10",
        ]

        if d.get("scenario_bear_prob"):
            lines += [
                "",
                "-- MONTE CARLO SCENARIOS (12M) --",
                f"Bear ({pct(d.get('scenario_bear_prob'))}): {pct(d.get('scenario_bear_return'))}",
                f"Base ({pct(d.get('scenario_base_prob'))}): {pct(d.get('scenario_base_return'))}",
                f"Bull ({pct(d.get('scenario_bull_prob'))}): {pct(d.get('scenario_bull_return'))}",
                f"Expected Value: {pct(d.get('scenario_expected_value'))}",
            ]

        return "\n".join(lines)

    @staticmethod
    def _json_schema_instructions(schema: dict) -> str:
        """Format expected JSON schema for the LLM."""
        return f"Respond with exactly this JSON structure:\n{json.dumps(schema, indent=2)}"
