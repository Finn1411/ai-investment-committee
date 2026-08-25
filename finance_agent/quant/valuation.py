"""
Valuation models — DCF, reverse-DCF, sensitivity analysis, historical percentiles.
All deterministic calculations. No LLM involvement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from finance_agent.quant.engine import QuantEngine
from finance_agent.utils.logger import logger


# ── DCF Result containers ─────────────────────────────────────────────────────

@dataclass
class DCFResult:
    intrinsic_value_per_share: float
    current_price: float
    margin_of_safety: float          # (intrinsic - price) / intrinsic
    upside_downside: float           # (intrinsic - price) / price
    implied_growth_rate: float       # growth rate used
    terminal_growth: float
    discount_rate: float
    forecast_years: int
    total_enterprise_value: float
    equity_value: float
    shares_outstanding: float


@dataclass
class ReverseDCFResult:
    implied_fcf_growth: float        # Growth rate the market is pricing in
    current_ev: float
    fcf_base: float
    terminal_growth: float
    discount_rate: float
    narrative: str                   # e.g. "Market prices in 18% FCF CAGR"


@dataclass
class SensitivityTable:
    """DCF intrinsic value across growth rate × discount rate grid."""
    growth_rates: list[float]
    discount_rates: list[float]
    values: list[list[float]]        # values[i][j] = intrinsic at growth[i], discount[j]

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        df = pd.DataFrame(
            self.values,
            index=[f"{g:.0%}" for g in self.growth_rates],
            columns=[f"{d:.0%}" for d in self.discount_rates],
        )
        df.index.name = "FCF Growth"
        df.columns.name = "Discount Rate"
        return df


@dataclass
class ValuationSummary:
    ticker: str
    current_price: float
    dcf: Optional[DCFResult] = None
    reverse_dcf: Optional[ReverseDCFResult] = None
    sensitivity: Optional[SensitivityTable] = None
    pe_percentile: Optional[float] = None
    ev_ebitda_percentile: Optional[float] = None
    ps_percentile: Optional[float] = None
    valuation_label: str = "UNKNOWN"   # "CHEAP", "FAIR", "EXPENSIVE", "VERY_EXPENSIVE"
    composite_valuation_score: Optional[float] = None  # 0-10, higher = cheaper


# ── Main valuation engine ─────────────────────────────────────────────────────

class ValuationEngine:
    """
    Builds a complete valuation picture from raw data + MetricSet.
    """

    DEFAULT_DISCOUNT_RATE = 0.10   # 10% WACC
    DEFAULT_TERMINAL_GROWTH = 0.03  # 3% terminal growth

    def build(
        self,
        ticker: str,
        current_price: float,
        fcf: float,
        shares_outstanding: float,
        enterprise_value: float,
        net_debt: float,
        # Optional DCF inputs
        growth_rate: Optional[float] = None,
        terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
        discount_rate: float = DEFAULT_DISCOUNT_RATE,
        forecast_years: int = 10,
        # Historical percentile inputs
        pe_percentile: Optional[float] = None,
        ev_ebitda_percentile: Optional[float] = None,
        ps_percentile: Optional[float] = None,
    ) -> ValuationSummary:
        """
        Build a complete ValuationSummary.

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            fcf: Trailing twelve month free cash flow (total $)
            shares_outstanding: Total diluted shares
            enterprise_value: Current market EV
            net_debt: Total debt - cash
            growth_rate: FCF growth for DCF (if None, tries to auto-estimate)
            terminal_growth: Terminal/perpetuity growth rate
            discount_rate: WACC / required return
            forecast_years: DCF forecast horizon
            pe/ev/ps percentile: pre-computed historical percentile (0-100)
        """
        summary = ValuationSummary(
            ticker=ticker,
            current_price=current_price,
            pe_percentile=pe_percentile,
            ev_ebitda_percentile=ev_ebitda_percentile,
            ps_percentile=ps_percentile,
        )

        if fcf > 0 and shares_outstanding > 0:
            # Auto-estimate growth if not provided
            if growth_rate is None:
                growth_rate = self._estimate_growth(enterprise_value, fcf, terminal_growth, discount_rate, forecast_years)

            summary.dcf = self._run_dcf(
                ticker=ticker,
                current_price=current_price,
                fcf=fcf,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
                growth_rate=growth_rate,
                terminal_growth=terminal_growth,
                discount_rate=discount_rate,
                forecast_years=forecast_years,
            )

            summary.reverse_dcf = self._run_reverse_dcf(
                current_ev=enterprise_value,
                fcf=fcf,
                terminal_growth=terminal_growth,
                discount_rate=discount_rate,
                forecast_years=forecast_years,
            )

            summary.sensitivity = self._build_sensitivity(
                fcf=fcf,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
                terminal_growth=terminal_growth,
                forecast_years=forecast_years,
            )

        summary.valuation_label = self._label(summary)
        summary.composite_valuation_score = self._composite_score(summary)
        return summary

    # ── DCF ───────────────────────────────────────────────────────────────────

    def _run_dcf(
        self,
        ticker: str,
        current_price: float,
        fcf: float,
        shares_outstanding: float,
        net_debt: float,
        growth_rate: float,
        terminal_growth: float,
        discount_rate: float,
        forecast_years: int,
    ) -> DCFResult:
        try:
            total_ev = QuantEngine.simple_dcf(
                fcf_base=fcf,
                growth_rate=growth_rate,
                terminal_growth=terminal_growth,
                discount_rate=discount_rate,
                forecast_years=forecast_years,
            )
            equity_value = max(total_ev - net_debt, 0)
            intrinsic_per_share = equity_value / shares_outstanding
            mos = (intrinsic_per_share - current_price) / intrinsic_per_share if intrinsic_per_share > 0 else -1.0
            upside = (intrinsic_per_share - current_price) / current_price

            logger.debug(
                f"[Valuation] {ticker} DCF: intrinsic={intrinsic_per_share:.2f}, "
                f"price={current_price:.2f}, MoS={mos:.1%}"
            )
            return DCFResult(
                intrinsic_value_per_share=round(intrinsic_per_share, 2),
                current_price=current_price,
                margin_of_safety=round(mos, 4),
                upside_downside=round(upside, 4),
                implied_growth_rate=growth_rate,
                terminal_growth=terminal_growth,
                discount_rate=discount_rate,
                forecast_years=forecast_years,
                total_enterprise_value=total_ev,
                equity_value=equity_value,
                shares_outstanding=shares_outstanding,
            )
        except Exception as e:
            logger.warning(f"[Valuation] DCF failed: {e}")
            return None

    def _run_reverse_dcf(
        self,
        current_ev: float,
        fcf: float,
        terminal_growth: float,
        discount_rate: float,
        forecast_years: int,
    ) -> Optional[ReverseDCFResult]:
        try:
            implied_growth = QuantEngine.reverse_dcf_growth(
                current_ev=current_ev,
                fcf_base=fcf,
                terminal_growth=terminal_growth,
                discount_rate=discount_rate,
                forecast_years=forecast_years,
            )
            narrative = (
                f"Market prices in {implied_growth:.1%} FCF CAGR over {forecast_years}y "
                f"(assuming {discount_rate:.0%} discount rate, "
                f"{terminal_growth:.1%} terminal growth)"
            )
            return ReverseDCFResult(
                implied_fcf_growth=round(implied_growth, 4),
                current_ev=current_ev,
                fcf_base=fcf,
                terminal_growth=terminal_growth,
                discount_rate=discount_rate,
                narrative=narrative,
            )
        except Exception as e:
            logger.debug(f"[Valuation] Reverse DCF failed: {e}")
            return None

    def _build_sensitivity(
        self,
        fcf: float,
        shares_outstanding: float,
        net_debt: float,
        terminal_growth: float,
        forecast_years: int,
    ) -> SensitivityTable:
        growth_rates = [-0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        discount_rates = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]

        values = []
        for g in growth_rates:
            row = []
            for d in discount_rates:
                try:
                    total_ev = QuantEngine.simple_dcf(fcf, g, terminal_growth, d, forecast_years)
                    equity = max(total_ev - net_debt, 0)
                    per_share = round(equity / shares_outstanding, 2) if shares_outstanding > 0 else 0
                    row.append(per_share)
                except Exception:
                    row.append(0.0)
            values.append(row)

        return SensitivityTable(
            growth_rates=growth_rates,
            discount_rates=discount_rates,
            values=values,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _estimate_growth(
        self,
        ev: float,
        fcf: float,
        terminal_growth: float,
        discount_rate: float,
        forecast_years: int,
    ) -> float:
        """Estimate a reasonable growth assumption from reverse-DCF, clipped to realistic range."""
        try:
            implied = QuantEngine.reverse_dcf_growth(ev, fcf, terminal_growth, discount_rate, forecast_years)
            # Use slightly below implied (conservative)
            return max(min(implied * 0.85, 0.30), -0.10)
        except Exception:
            return 0.07  # Default 7% if calculation fails

    def _label(self, s: ValuationSummary) -> str:
        """Simple label based on DCF margin of safety and historical percentiles."""
        signals = []

        if s.dcf:
            mos = s.dcf.margin_of_safety
            if mos > 0.30:
                signals.append("CHEAP")
            elif mos > 0.10:
                signals.append("FAIR")
            elif mos > -0.10:
                signals.append("SLIGHTLY_EXPENSIVE")
            else:
                signals.append("EXPENSIVE")

        if s.pe_percentile is not None:
            if s.pe_percentile > 85:
                signals.append("EXPENSIVE")
            elif s.pe_percentile < 30:
                signals.append("CHEAP")

        if not signals:
            return "UNKNOWN"

        # Majority vote
        cheap = signals.count("CHEAP")
        expensive = signals.count("EXPENSIVE") + signals.count("SLIGHTLY_EXPENSIVE")
        if cheap > expensive:
            return "CHEAP"
        elif expensive > cheap:
            return "EXPENSIVE" if signals.count("EXPENSIVE") >= signals.count("SLIGHTLY_EXPENSIVE") else "SLIGHTLY_EXPENSIVE"
        return "FAIR"

    def _composite_score(self, s: ValuationSummary) -> Optional[float]:
        """
        0–10 composite valuation score.
        10 = deeply undervalued, 0 = extremely overvalued.
        """
        scores = []

        # DCF margin of safety
        if s.dcf:
            mos = s.dcf.margin_of_safety
            # MoS 50% → 10, 0% → 5, -50% → 0
            scores.append(max(0.0, min(10.0, (mos + 0.5) * 10)))

        # Valuation percentile (lower percentile = cheaper = higher score)
        for pct in [s.pe_percentile, s.ev_ebitda_percentile, s.ps_percentile]:
            if pct is not None:
                scores.append(10.0 - (pct / 10.0))  # 100th pctile → 0, 0th → 10

        if not scores:
            return None
        return round(sum(scores) / len(scores), 2)
