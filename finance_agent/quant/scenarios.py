"""
Bull/Base/Bear scenario builder.
Uses Monte Carlo simulation + fundamental assumptions to produce
structured ScenarioModel objects ready for the Committee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from finance_agent.models.schemas import Horizon, Scenario, ScenarioModel
from finance_agent.quant.engine import QuantEngine
from finance_agent.quant.metrics import MetricSet
from finance_agent.utils.logger import logger


@dataclass
class ScenarioInputs:
    """
    All inputs needed to build a scenario model.
    Populated from MetricSet + optional analyst overrides.
    """
    ticker: str
    horizon: Horizon

    # Price / market
    current_price: float
    market_cap: Optional[float] = None
    beta: Optional[float] = None

    # Historical risk
    volatility_annual: Optional[float] = None   # e.g. 0.25 for 25%
    max_drawdown_1y: Optional[float] = None

    # Growth
    revenue_growth_yoy: Optional[float] = None
    eps_growth_yoy: Optional[float] = None
    fcf_growth_yoy: Optional[float] = None

    # Quality
    roic: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    piotroski_f_score: Optional[int] = None

    # Valuation
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pe_percentile_5y: Optional[float] = None
    fcf_yield: Optional[float] = None

    # Monte Carlo params
    expected_annual_return: Optional[float] = None   # If None, auto-estimated
    n_simulations: int = 10_000
    seed: int = 42

    # Analyst overrides — set these to skip auto-estimation
    bear_return_override: Optional[float] = None
    base_return_override: Optional[float] = None
    bull_return_override: Optional[float] = None


@dataclass
class ScenarioBuildResult:
    model: ScenarioModel
    monte_carlo_used: bool
    expected_annual_return_used: float
    volatility_used: float
    bear_threshold: float
    bull_threshold: float
    inputs: ScenarioInputs


class ScenarioBuilder:
    """
    Builds probabilistic Bull/Base/Bear scenarios.

    Strategy:
    1. Estimate expected return from fundamentals (FCF yield + growth)
    2. Use historical volatility for the Monte Carlo distribution
    3. Calibrate Bear/Bull thresholds dynamically (1 StdDev)
    4. Derive scenario narratives from fundamentals
    """

    # Benchmark expected return assumptions
    BENCHMARK_ANNUAL_RETURN = 0.09    # ~9% historical S&P 500

    def build(self, inputs: ScenarioInputs) -> ScenarioBuildResult:
        logger.info(f"[ScenarioBuilder] Building scenarios for {inputs.ticker} | {inputs.horizon.value}")

        # ── Step 1: Estimate expected return ─────────────────────────────────
        exp_return = inputs.expected_annual_return or self._estimate_expected_return(inputs)

        # ── Step 2: Volatility ────────────────────────────────────────────────
        vol = inputs.volatility_annual or self._estimate_volatility(inputs)

        # ── Step 3: Horizon in years ──────────────────────────────────────────
        horizon_years = self._horizon_to_years(inputs.horizon)

        # ── Step 4: Thresholds (dynamic: ±1 StdDev over horizon) ─────────────
        horizon_vol = vol * math.sqrt(horizon_years)
        bear_threshold = round(exp_return * horizon_years - horizon_vol, 2)
        bull_threshold = round(exp_return * horizon_years + horizon_vol, 2)

        # Clamp to reasonable bounds
        bear_threshold = max(bear_threshold, -0.70)
        bull_threshold = min(bull_threshold, 2.00)

        # ── Step 5: Monte Carlo ───────────────────────────────────────────────
        mc_returns = QuantEngine.monte_carlo_returns(
            expected_annual_return=exp_return,
            annual_volatility=vol,
            horizon_years=horizon_years,
            n_simulations=inputs.n_simulations,
            seed=inputs.seed,
        )
        probs = QuantEngine.scenario_probabilities_from_mc(
            mc_returns,
            bear_threshold=bear_threshold,
            bull_threshold=bull_threshold,
        )

        # ── Step 6: Override returns if analyst provided them ─────────────────
        bear_return = inputs.bear_return_override or probs["bear_median_return"]
        base_return = inputs.base_return_override or probs["base_median_return"]
        bull_return = inputs.bull_return_override or probs["bull_median_return"]

        # ── Step 7: Build narratives ──────────────────────────────────────────
        bear_narrative = self._bear_narrative(inputs)
        base_narrative = self._base_narrative(inputs, exp_return)
        bull_narrative = self._bull_narrative(inputs)

        # ── Step 8: Assemble model ────────────────────────────────────────────
        model = ScenarioModel(
            ticker=inputs.ticker,
            horizon=inputs.horizon,
            analysis_date=date.today(),
            bear=Scenario(
                label="Bear",
                probability=probs["bear_prob"],
                expected_return=round(bear_return, 3),
                narrative=bear_narrative,
            ),
            base=Scenario(
                label="Base",
                probability=probs["base_prob"],
                expected_return=round(base_return, 3),
                narrative=base_narrative,
            ),
            bull=Scenario(
                label="Bull",
                probability=probs["bull_prob"],
                expected_return=round(bull_return, 3),
                narrative=bull_narrative,
            ),
        )

        logger.info(
            f"[ScenarioBuilder] {inputs.ticker} | "
            f"EV={model.expected_value:.1%} | "
            f"Bear={model.bear.probability:.0%} / "
            f"Base={model.base.probability:.0%} / "
            f"Bull={model.bull.probability:.0%}"
        )

        return ScenarioBuildResult(
            model=model,
            monte_carlo_used=True,
            expected_annual_return_used=exp_return,
            volatility_used=vol,
            bear_threshold=bear_threshold,
            bull_threshold=bull_threshold,
            inputs=inputs,
        )

    # ── Return estimation ─────────────────────────────────────────────────────

    def _estimate_expected_return(self, inp: ScenarioInputs) -> float:
        """
        Estimate expected annual return using:
        - FCF yield (income component)
        - Growth rate (earnings growth component)
        - Valuation multiple change (re-rating component, conservative)
        """
        components = []

        # 1. FCF yield as base income return
        if inp.fcf_yield and inp.fcf_yield > 0:
            components.append(inp.fcf_yield)

        # 2. Growth component (use FCF growth, fall back to revenue/EPS)
        growth = inp.fcf_growth_yoy or inp.eps_growth_yoy or inp.revenue_growth_yoy
        if growth is not None:
            # Conservative: use 60% of observed growth (mean reversion)
            components.append(growth * 0.60)

        # 3. Valuation mean reversion (if richly valued, drag on return)
        if inp.pe_percentile_5y is not None:
            # Rich valuation → negative drag, cheap → positive tailwind
            mean_reversion_drag = -(inp.pe_percentile_5y - 50) / 100 * 0.03
            components.append(mean_reversion_drag)

        if components:
            estimated = sum(components)
            # Clamp: -20% to +40% annual
            return max(-0.20, min(0.40, estimated))

        # Fallback: ROIC-based (Buffett: return ≈ ROIC if retained earnings reinvested well)
        if inp.roic and inp.roic > 0:
            return max(-0.20, min(0.40, inp.roic * 0.7))

        # Last resort: use benchmark return
        logger.debug(f"[ScenarioBuilder] {inp.ticker}: using benchmark return as fallback")
        return self.BENCHMARK_ANNUAL_RETURN

    def _estimate_volatility(self, inp: ScenarioInputs) -> float:
        """Estimate annualised volatility from available data."""
        if inp.volatility_annual and 0 < inp.volatility_annual < 2.0:
            return inp.volatility_annual

        # Rough estimate from beta
        MARKET_VOL = 0.17  # ~17% historical S&P 500 volatility
        if inp.beta:
            return max(0.10, min(1.20, abs(inp.beta) * MARKET_VOL))

        # Sector-agnostic fallback
        return 0.25

    @staticmethod
    def _horizon_to_years(horizon: Horizon) -> float:
        return {"3M": 0.25, "12M": 1.0, "3-5Y": 4.0}.get(horizon.value, 1.0)

    # ── Narrative generation ──────────────────────────────────────────────────

    def _bear_narrative(self, inp: ScenarioInputs) -> str:
        risks = []
        if inp.net_debt_to_ebitda and inp.net_debt_to_ebitda > 3:
            risks.append("high leverage constrains flexibility")
        if inp.pe_percentile_5y and inp.pe_percentile_5y > 80:
            risks.append("valuation compression from rich multiples")
        if inp.revenue_growth_yoy and inp.revenue_growth_yoy < 0:
            risks.append("revenue contraction accelerates")
        if not risks:
            risks.append("macro deterioration or sector rotation")
        return f"Bear case: {'; '.join(risks)}."

    def _base_narrative(self, inp: ScenarioInputs, exp_return: float) -> str:
        return (
            f"Base case: company executes on current trajectory, "
            f"delivering ~{exp_return:.0%} annualised return "
            f"driven by {'FCF yield + ' if inp.fcf_yield else ''}"
            f"{'growth' if inp.revenue_growth_yoy else 'stable operations'}."
        )

    def _bull_narrative(self, inp: ScenarioInputs) -> str:
        catalysts = []
        if inp.roic and inp.roic > 0.20:
            catalysts.append("high ROIC compounds at above-average reinvestment rates")
        if inp.fcf_growth_yoy and inp.fcf_growth_yoy > 0.20:
            catalysts.append("FCF growth re-accelerates")
        if inp.pe_percentile_5y and inp.pe_percentile_5y < 40:
            catalysts.append("multiple re-rating as cheap valuation recognized")
        if not catalysts:
            catalysts.append("above-consensus execution and market re-rating")
        return f"Bull case: {'; '.join(catalysts)}."


# ── Convenience function ──────────────────────────────────────────────────────

def build_scenarios_from_metrics(
    ticker: str,
    metrics: MetricSet,
    horizon: Horizon = Horizon.TWELVE_MONTHS,
    current_price: Optional[float] = None,
    **overrides,
) -> ScenarioBuildResult:
    """
    Convenience wrapper: builds ScenarioInputs from a MetricSet then runs the builder.
    """
    inputs = ScenarioInputs(
        ticker=ticker,
        horizon=horizon,
        current_price=current_price or metrics.get("current_price", 0),
        market_cap=metrics.get("market_cap"),
        beta=metrics.get("beta"),
        volatility_annual=metrics.get("volatility_90d"),
        max_drawdown_1y=metrics.get("max_drawdown_1y"),
        revenue_growth_yoy=metrics.get("revenue_growth_yoy"),
        eps_growth_yoy=metrics.get("eps_growth_yoy"),
        fcf_growth_yoy=metrics.get("fcf_growth_yoy"),
        roic=metrics.get("roic"),
        net_debt_to_ebitda=metrics.get("net_debt_to_ebitda"),
        piotroski_f_score=metrics.get("piotroski_f_score"),
        pe_ratio=metrics.get("pe_ratio"),
        forward_pe=metrics.get("forward_pe"),
        pe_percentile_5y=metrics.get("pe_percentile_5y"),
        fcf_yield=metrics.get("fcf_yield"),
        **overrides,
    )
    return ScenarioBuilder().build(inputs)
