"""
Quant Engine — all deterministic financial calculations.
LLMs never touch these numbers directly; they only interpret the results.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from finance_agent.utils.logger import logger


class QuantEngine:
    """
    Pure calculation class. No I/O, no LLM calls.
    Feed it price series / fundamental data; get back validated metrics.
    """

    # ── Returns ───────────────────────────────────────────────────────────────

    @staticmethod
    def simple_return(price_start: float, price_end: float) -> float:
        """Simple period return."""
        if price_start <= 0:
            raise ValueError("price_start must be positive")
        return (price_end - price_start) / price_start

    @staticmethod
    def log_return(price_start: float, price_end: float) -> float:
        """Continuous (log) return."""
        if price_start <= 0 or price_end <= 0:
            raise ValueError("Prices must be positive")
        return math.log(price_end / price_start)

    @staticmethod
    def cagr(price_start: float, price_end: float, years: float) -> float:
        """Compound Annual Growth Rate."""
        if years <= 0:
            raise ValueError("years must be positive")
        if price_start <= 0:
            raise ValueError("price_start must be positive")
        return (price_end / price_start) ** (1 / years) - 1

    @staticmethod
    def cagr_from_series(
        values: list[float] | np.ndarray, periods_per_year: int = 1
    ) -> float:
        """CAGR from a sequence of values (e.g., annual revenues)."""
        arr = np.array(values, dtype=float)
        if len(arr) < 2 or arr[0] <= 0:
            raise ValueError("Need at least 2 positive values")
        n_years = (len(arr) - 1) / periods_per_year
        return (arr[-1] / arr[0]) ** (1 / n_years) - 1

    # ── Risk Metrics ──────────────────────────────────────────────────────────

    @staticmethod
    def annualised_volatility(
        daily_returns: pd.Series | np.ndarray, trading_days: int = 252
    ) -> float:
        """Annualised standard deviation of daily returns."""
        arr = np.array(daily_returns, dtype=float)
        return float(np.std(arr, ddof=1) * math.sqrt(trading_days))

    @staticmethod
    def maximum_drawdown(prices: pd.Series | np.ndarray) -> float:
        """Maximum drawdown from peak (negative number, e.g. -0.35)."""
        arr = np.array(prices, dtype=float)
        peak = np.maximum.accumulate(arr)
        drawdowns = (arr - peak) / peak
        return float(np.min(drawdowns))

    @staticmethod
    def beta(
        asset_returns: pd.Series | np.ndarray,
        benchmark_returns: pd.Series | np.ndarray,
    ) -> float:
        """Market beta of asset relative to benchmark."""
        x = np.array(benchmark_returns, dtype=float)
        y = np.array(asset_returns, dtype=float)
        if len(x) != len(y):
            raise ValueError("asset_returns and benchmark_returns must have the same length")
        cov = np.cov(y, x, ddof=1)
        var_bench = cov[1, 1]
        if var_bench == 0:
            return float("nan")
        return float(cov[0, 1] / var_bench)

    @staticmethod
    def sharpe_ratio(
        daily_returns: pd.Series | np.ndarray,
        risk_free_annual: float = 0.04,
        trading_days: int = 252,
    ) -> float:
        """Annualised Sharpe Ratio."""
        arr = np.array(daily_returns, dtype=float)
        rf_daily = (1 + risk_free_annual) ** (1 / trading_days) - 1
        excess = arr - rf_daily
        std = np.std(excess, ddof=1)
        if std == 0:
            return float("nan")
        return float(np.mean(excess) / std * math.sqrt(trading_days))

    @staticmethod
    def sortino_ratio(
        daily_returns: pd.Series | np.ndarray,
        risk_free_annual: float = 0.04,
        trading_days: int = 252,
        target_return: float = 0.0,
    ) -> float:
        """Annualised Sortino Ratio (only downside deviation)."""
        arr = np.array(daily_returns, dtype=float)
        rf_daily = (1 + risk_free_annual) ** (1 / trading_days) - 1
        excess = arr - rf_daily
        downside = excess[excess < target_return]
        if len(downside) == 0:
            return float("inf")
        downside_std = math.sqrt(np.mean(downside**2))
        if downside_std == 0:
            return float("nan")
        return float(np.mean(excess) / downside_std * math.sqrt(trading_days))

    @staticmethod
    def value_at_risk(
        daily_returns: pd.Series | np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """Historical VaR at given confidence level (negative number)."""
        arr = np.array(daily_returns, dtype=float)
        return float(np.percentile(arr, (1 - confidence) * 100))

    @staticmethod
    def relative_strength(
        asset_returns: pd.Series | np.ndarray,
        benchmark_returns: pd.Series | np.ndarray,
    ) -> float:
        """Simple relative strength: cumulative asset return / benchmark return."""
        a = (1 + np.array(asset_returns, dtype=float)).prod()
        b = (1 + np.array(benchmark_returns, dtype=float)).prod()
        if b == 0:
            return float("nan")
        return float(a / b)

    # ── Valuation ─────────────────────────────────────────────────────────────

    @staticmethod
    def historical_percentile(
        current_value: float,
        historical_values: list[float] | np.ndarray,
    ) -> float:
        """Where does current_value sit in its own history? Returns 0-100."""
        arr = np.array(historical_values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return float("nan")
        return float(np.mean(arr <= current_value) * 100)

    @staticmethod
    def fcf_yield(free_cash_flow: float, market_cap: float) -> float:
        if market_cap <= 0:
            raise ValueError("market_cap must be positive")
        return free_cash_flow / market_cap

    @staticmethod
    def ev(market_cap: float, total_debt: float, cash: float) -> float:
        """Enterprise Value = Market Cap + Debt - Cash."""
        return market_cap + total_debt - cash

    @staticmethod
    def ev_to_ebitda(enterprise_value: float, ebitda: float) -> Optional[float]:
        if ebitda <= 0:
            return None
        return enterprise_value / ebitda

    # ── DCF (simplified) ──────────────────────────────────────────────────────

    @staticmethod
    def simple_dcf(
        fcf_base: float,
        growth_rate: float,
        terminal_growth: float,
        discount_rate: float,
        forecast_years: int = 10,
    ) -> float:
        """
        Two-stage DCF.
        Returns intrinsic value (enterprise value proxy based on FCF).
        """
        if discount_rate <= terminal_growth:
            raise ValueError("discount_rate must exceed terminal_growth")
        pv = 0.0
        fcf = fcf_base
        for yr in range(1, forecast_years + 1):
            fcf *= (1 + growth_rate)
            pv += fcf / (1 + discount_rate) ** yr
        # Terminal value (Gordon Growth)
        terminal_fcf = fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv += terminal_value / (1 + discount_rate) ** forecast_years
        return pv

    @staticmethod
    def reverse_dcf_growth(
        current_ev: float,
        fcf_base: float,
        terminal_growth: float,
        discount_rate: float,
        forecast_years: int = 10,
        tolerance: float = 1e-4,
        max_iter: int = 100,
    ) -> float:
        """
        Reverse DCF: what FCF growth rate is implied by the current EV?
        Uses binary search.
        """
        low, high = -0.5, 2.0
        for _ in range(max_iter):
            mid = (low + high) / 2
            try:
                val = QuantEngine.simple_dcf(
                    fcf_base, mid, terminal_growth, discount_rate, forecast_years
                )
            except ValueError:
                high = mid
                continue
            if abs(val - current_ev) < tolerance * current_ev:
                return mid
            if val < current_ev:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    # ── Monte Carlo ───────────────────────────────────────────────────────────

    @staticmethod
    def monte_carlo_returns(
        expected_annual_return: float,
        annual_volatility: float,
        horizon_years: float = 1.0,
        n_simulations: int = 10_000,
        seed: int = 42,
    ) -> np.ndarray:
        """
        GBM Monte Carlo — returns array of simulated total returns.
        """
        rng = np.random.default_rng(seed)
        trading_days = int(252 * horizon_years)
        daily_mu = expected_annual_return / 252
        daily_sigma = annual_volatility / math.sqrt(252)
        daily_r = rng.normal(daily_mu, daily_sigma, (n_simulations, trading_days))
        total_returns = np.prod(1 + daily_r, axis=1) - 1
        return total_returns

    @staticmethod
    def scenario_probabilities_from_mc(
        mc_returns: np.ndarray,
        bear_threshold: float = -0.15,
        bull_threshold: float = 0.25,
    ) -> dict[str, float]:
        """
        Derive bear/base/bull probabilities from Monte Carlo distribution.
        """
        n = len(mc_returns)
        bear = float(np.mean(mc_returns <= bear_threshold))
        bull = float(np.mean(mc_returns >= bull_threshold))
        base = 1.0 - bear - bull
        return {
            "bear_prob": round(bear, 4),
            "base_prob": round(max(base, 0.0), 4),
            "bull_prob": round(bull, 4),
            "bear_median_return": float(np.median(mc_returns[mc_returns <= bear_threshold])) if bear > 0 else bear_threshold,
            "base_median_return": float(np.median(mc_returns[(mc_returns > bear_threshold) & (mc_returns < bull_threshold)])) if base > 0 else 0.05,
            "bull_median_return": float(np.median(mc_returns[mc_returns >= bull_threshold])) if bull > 0 else bull_threshold,
        }

    # ── Institutional Risk Metrics ─────────────────────────────────────────────

    @staticmethod
    def calmar_ratio(cagr: float, max_drawdown: float) -> Optional[float]:
        """Calmar Ratio = CAGR / |Max Drawdown|. Higher = better risk-adj return."""
        if max_drawdown == 0:
            return None
        return float(cagr / abs(max_drawdown))

    @staticmethod
    def omega_ratio(
        daily_returns: pd.Series | np.ndarray,
        threshold: float = 0.0,
    ) -> float:
        """
        Omega Ratio = sum(gains above threshold) / sum(losses below threshold).
        Threshold is a daily return. Values > 1 are desirable.
        """
        arr = np.array(daily_returns, dtype=float)
        gains  = arr[arr > threshold] - threshold
        losses = threshold - arr[arr <= threshold]
        if losses.sum() == 0:
            return float("inf")
        return float(gains.sum() / losses.sum())

    @staticmethod
    def conditional_var(
        daily_returns: pd.Series | np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        Conditional VaR (CVaR / Expected Shortfall) at given confidence.
        Returns the expected loss in the worst (1-confidence) fraction of days.
        Result is negative (e.g. -0.03 = -3% average daily loss in tail).
        """
        arr = np.array(daily_returns, dtype=float)
        var_threshold = np.percentile(arr, (1 - confidence) * 100)
        tail = arr[arr <= var_threshold]
        return float(np.mean(tail)) if len(tail) > 0 else float(var_threshold)

    @staticmethod
    def information_ratio(
        portfolio_returns: pd.Series | np.ndarray,
        benchmark_returns: pd.Series | np.ndarray,
    ) -> Optional[float]:
        """
        Information Ratio = annualised alpha / annualised tracking error.
        Measures manager skill (or factor alpha) per unit of active risk.
        """
        p = np.array(portfolio_returns, dtype=float)
        b = np.array(benchmark_returns, dtype=float)
        n = min(len(p), len(b))
        if n < 20:
            return None
        excess = p[:n] - b[:n]
        te = float(np.std(excess, ddof=1)) * math.sqrt(252)
        alpha = float(np.mean(excess)) * 252
        if te == 0:
            return None
        return float(alpha / te)

    @staticmethod
    def treynor_ratio(
        daily_returns: pd.Series | np.ndarray,
        beta: float,
        risk_free_annual: float = 0.04,
        trading_days: int = 252,
    ) -> Optional[float]:
        """
        Treynor Ratio = (Annualised Return - Rf) / Beta.
        Measures systematic-risk-adjusted excess return.
        """
        if beta == 0 or math.isnan(beta):
            return None
        arr = np.array(daily_returns, dtype=float)
        annual_ret = float((1 + np.mean(arr)) ** trading_days - 1)
        return float((annual_ret - risk_free_annual) / beta)

    @staticmethod
    def rsi(
        prices: pd.Series | np.ndarray,
        period: int = 14,
    ) -> Optional[float]:
        """
        Wilder's Relative Strength Index (0–100).
        >70 = overbought, <30 = oversold.
        Uses exponential smoothing per Wilder (1978), not simple moving average.
        """
        arr = np.array(prices, dtype=float)
        if len(arr) < period + 1:
            return None
        deltas = np.diff(arr)
        gains  = np.where(deltas > 0,  deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        if len(gains) < period:
            return None

        # Seed with simple averages
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))

        # Wilder smoothing (EMA with α = 1/period)
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(float(100 - 100 / (1 + rs)), 2)

    @staticmethod
    def z_score_normalise(values: list) -> list:
        """
        Cross-sectional z-score normalisation for factor ranking.
        None values are preserved as None. Result: mean=0, std≈1 over valid values.
        Used to compare stocks within a sector on the same scale.
        """
        valid = [
            v for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        if len(valid) < 2:
            return [None] * len(values)
        mu    = float(np.mean(valid))
        sigma = float(np.std(valid, ddof=1))
        if sigma == 0:
            return [0.0 if v is not None else None for v in values]
        return [
            round(float((v - mu) / sigma), 4) if v is not None else None
            for v in values
        ]
