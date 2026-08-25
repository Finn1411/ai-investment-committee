"""
Full 40+ metric computation from raw yfinance data.
All calculations are deterministic — no LLM involved.

Metric categories:
  1. Market / Price metrics      (15 metrics)
  2. Fundamental quality         (12 metrics)
  3. Growth                      (6 metrics)
  4. Valuation                   (8 metrics)
  5. Composite scores            (4 scores)
  Total: 45 metrics
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from finance_agent.data.fetcher import RawTickerData, YFinanceFetcher
from finance_agent.quant.engine import QuantEngine
from finance_agent.utils.logger import logger


# ── Result container ──────────────────────────────────────────────────────────

class MetricSet:
    """
    Holds all computed metrics as a flat dict plus typed attributes.
    Access via .metrics dict or directly e.g. ms.pe_ratio
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        return self._data.get(key)

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._data)

    @property
    def available_count(self) -> int:
        return sum(1 for v in self._data.values() if v is not None)

    def summary(self) -> str:
        lines = [f"  {k}: {v}" for k, v in sorted(self._data.items()) if v is not None]
        return "\n".join(lines)


# ── Main metrics engine ───────────────────────────────────────────────────────

class MetricsEngine:
    """
    Computes all 45 metrics from RawTickerData.
    Uses QuantEngine for all math.
    """

    def __init__(self, benchmark_ticker: str = "^GSPC") -> None:
        self.benchmark_ticker = benchmark_ticker
        self._fetcher = YFinanceFetcher()
        self._qe = QuantEngine()

    def compute(self, raw: RawTickerData, benchmark_prices: Optional[pd.DataFrame] = None) -> MetricSet:
        """
        Main entry point. Returns MetricSet with all available metrics.

        Args:
            raw: Output from YFinanceFetcher.fetch()
            benchmark_prices: Optional pre-fetched benchmark OHLCV (to avoid extra API call)
        """
        ms = MetricSet()
        logger.info(f"[MetricsEngine] Computing metrics for {raw.ticker}")

        # Fetch benchmark if not provided
        if benchmark_prices is None:
            try:
                bm_raw = self._fetcher.fetch(self.benchmark_ticker, price_period="2y")
                benchmark_prices = bm_raw.price_history
            except Exception as e:
                logger.warning(f"[MetricsEngine] Could not fetch benchmark: {e}")

        self._compute_market_metrics(ms, raw, benchmark_prices)
        self._compute_fundamental_metrics(ms, raw)
        self._compute_growth_metrics(ms, raw)
        self._compute_valuation_metrics(ms, raw)
        self._compute_composite_scores(ms, raw)

        logger.info(
            f"[MetricsEngine] {raw.ticker}: {ms.available_count}/45 metrics computed"
        )
        return ms

    # ── 1. Market / Price Metrics ─────────────────────────────────────────────

    def _compute_market_metrics(
        self,
        ms: MetricSet,
        raw: RawTickerData,
        benchmark: Optional[pd.DataFrame],
    ) -> None:
        prices = raw.price_history
        if prices is None or prices.empty:
            logger.warning(f"[MetricsEngine] {raw.ticker}: no price data")
            return

        close = prices["Close"].dropna()
        volume = prices["Volume"].dropna()
        daily_returns = close.pct_change().dropna()

        today = close.index[-1].date()

        # ── Period returns ────────────────────────────────────────────────────
        for label, delta_days in [
            ("return_1d", 1), ("return_1w", 5), ("return_1m", 21),
            ("return_3m", 63), ("return_6m", 126), ("return_1y", 252),
        ]:
            ms.set(label, self._period_return(close, delta_days))

        # YTD return
        ytd_start = pd.Timestamp(date(today.year, 1, 1))
        ytd_prices = close[close.index >= ytd_start]
        if len(ytd_prices) >= 2:
            ms.set("return_ytd", QuantEngine.simple_return(float(ytd_prices.iloc[0]), float(ytd_prices.iloc[-1])))

        # ── Volatility ────────────────────────────────────────────────────────
        if len(daily_returns) >= 30:
            ms.set("volatility_30d", QuantEngine.annualised_volatility(daily_returns.iloc[-30:]))
        if len(daily_returns) >= 90:
            ms.set("volatility_90d", QuantEngine.annualised_volatility(daily_returns.iloc[-90:]))

        # ── Risk-adjusted returns ─────────────────────────────────────────────
        if len(daily_returns) >= 252:
            ms.set("sharpe_ratio_1y", QuantEngine.sharpe_ratio(daily_returns.iloc[-252:]))
            ms.set("sortino_ratio_1y", QuantEngine.sortino_ratio(daily_returns.iloc[-252:]))
            ms.set("var_95_1y", QuantEngine.value_at_risk(daily_returns.iloc[-252:], 0.95))
            ms.set("max_drawdown_1y", QuantEngine.maximum_drawdown(close.iloc[-252:]))

        # ── Beta & relative strength ──────────────────────────────────────────
        if benchmark is not None and not benchmark.empty:
            bm_close = benchmark["Close"].dropna()
            bm_returns = bm_close.pct_change().dropna()

            # Align on common dates
            aligned = daily_returns.align(bm_returns, join="inner")
            asset_r, bm_r = aligned[0], aligned[1]

            if len(asset_r) >= 60:
                ms.set("beta", QuantEngine.beta(asset_r.values, bm_r.values))
                ms.set(
                    "relative_strength_vs_benchmark",
                    QuantEngine.relative_strength(asset_r.iloc[-252:].values, bm_r.iloc[-252:].values)
                    if len(asset_r) >= 252 else None,
                )

        # ── Volume & market cap ───────────────────────────────────────────────
        if len(volume) >= 30:
            ms.set("avg_daily_volume_30d", float(volume.iloc[-30:].mean()))

        info = raw.info
        ms.set("market_cap", info.get("marketCap"))
        ms.set("shares_outstanding", info.get("sharesOutstanding"))
        ms.set("current_price", float(close.iloc[-1]) if not close.empty else None)

    # ── 2. Fundamental Quality Metrics ───────────────────────────────────────

    def _compute_fundamental_metrics(self, ms: MetricSet, raw: RawTickerData) -> None:
        info = raw.info
        get = YFinanceFetcher.get_statement_row
        latest = YFinanceFetcher.latest_value

        # ── Profitability ─────────────────────────────────────────────────────
        revenue = latest(get(raw.income_stmt, "Total Revenue"))
        gross_profit = latest(get(raw.income_stmt, "Gross Profit"))
        ebit = latest(get(raw.income_stmt, "EBIT", "Operating Income"))
        net_income = latest(get(raw.income_stmt, "Net Income"))

        if revenue and revenue > 0:
            if gross_profit is not None:
                ms.set("gross_margin", gross_profit / revenue)
            if ebit is not None:
                ms.set("operating_margin", ebit / revenue)
            if net_income is not None:
                ms.set("net_margin", net_income / revenue)

        # Fallback to info dict
        if ms.get("gross_margin") is None:
            ms.set("gross_margin", info.get("grossMargins"))
        if ms.get("operating_margin") is None:
            ms.set("operating_margin", info.get("operatingMargins"))
        if ms.get("net_margin") is None:
            ms.set("net_margin", info.get("profitMargins"))

        # ── Cash Flow ─────────────────────────────────────────────────────────
        op_cf = latest(get(raw.cash_flow, "Operating Cash Flow", "Total Cash From Operating Activities"))
        capex = latest(get(raw.cash_flow, "Capital Expenditure", "Capital Expenditures"))

        # FCF = Operating CF - Capex (capex is usually negative in yfinance)
        if op_cf is not None and capex is not None:
            fcf = op_cf + capex  # capex is negative
            ms.set("free_cash_flow", fcf)
            if revenue and revenue > 0:
                ms.set("fcf_margin", fcf / revenue)
            market_cap = ms.get("market_cap")
            if market_cap and market_cap > 0:
                ms.set("fcf_yield", fcf / market_cap)
            # Cash conversion: FCF / Net Income
            if net_income and net_income != 0:
                ms.set("cash_conversion", fcf / net_income)
        else:
            # Try info fallback
            ms.set("free_cash_flow", info.get("freeCashflow"))

        # ── Balance Sheet Quality ─────────────────────────────────────────────
        total_debt = latest(get(raw.balance_sheet, "Total Debt", "Long Term Debt"))
        cash = latest(get(raw.balance_sheet, "Cash And Cash Equivalents", "Cash"))
        curr_assets = latest(get(raw.balance_sheet, "Current Assets", "Total Current Assets"))
        curr_liab = latest(get(raw.balance_sheet, "Current Liabilities", "Total Current Liabilities"))
        ebitda = info.get("ebitda")

        ms.set("total_debt", total_debt)
        ms.set("cash", cash)

        if total_debt is not None and cash is not None:
            ms.set("net_debt", total_debt - cash)

        if ebitda and ebitda > 0 and total_debt is not None and cash is not None:
            net_debt = total_debt - cash
            ms.set("net_debt_to_ebitda", net_debt / ebitda)

        # Interest coverage
        interest_exp = latest(get(raw.income_stmt, "Interest Expense"))
        if ebit is not None and interest_exp is not None and interest_exp != 0:
            ms.set("interest_coverage", ebit / abs(interest_exp))

        # Current ratio
        if curr_assets is not None and curr_liab and curr_liab > 0:
            ms.set("current_ratio", curr_assets / curr_liab)

        # ── Return metrics ────────────────────────────────────────────────────
        ms.set("roic", info.get("returnOnEquity"))  # yfinance doesn't expose ROIC directly
        ms.set("roe", info.get("returnOnEquity"))
        ms.set("roa", info.get("returnOnAssets"))

        # Better ROIC approximation: EBIT*(1-tax) / (Equity + Debt - Cash)
        total_equity = latest(get(raw.balance_sheet, "Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"))
        tax_rate = info.get("effectiveTaxRate", 0.21)
        if ebit and total_equity and total_debt and cash:
            invested_capital = total_equity + total_debt - cash
            if invested_capital > 0:
                nopat = ebit * (1 - tax_rate)
                ms.set("roic", nopat / invested_capital)

    # ── 3. Growth Metrics ─────────────────────────────────────────────────────

    def _compute_growth_metrics(self, ms: MetricSet, raw: RawTickerData) -> None:
        get = YFinanceFetcher.get_statement_row
        latest = YFinanceFetcher.latest_value
        prior = YFinanceFetcher.value_n_periods_ago

        income = raw.income_stmt

        # ── Revenue growth ────────────────────────────────────────────────────
        rev_row = get(income, "Total Revenue")
        rev_now = latest(rev_row)
        rev_prior = prior(rev_row, 1)
        if rev_now and rev_prior and rev_prior > 0:
            ms.set("revenue_growth_yoy", (rev_now - rev_prior) / abs(rev_prior))

        # 3Y CAGR
        rev_3y_ago = prior(rev_row, 3)
        if rev_now and rev_3y_ago and rev_3y_ago > 0:
            try:
                ms.set("revenue_cagr_3y", QuantEngine.cagr(rev_3y_ago, rev_now, 3))
            except Exception:
                pass

        # ── EPS growth ────────────────────────────────────────────────────────
        eps_row = get(income, "Diluted EPS", "Basic EPS", "EPS")
        eps_now = latest(eps_row)
        eps_prior = prior(eps_row, 1)
        if eps_now and eps_prior and eps_prior != 0:
            ms.set("eps_growth_yoy", (eps_now - eps_prior) / abs(eps_prior))

        eps_3y_ago = prior(eps_row, 3)
        if eps_now and eps_3y_ago and eps_3y_ago > 0:
            try:
                ms.set("eps_cagr_3y", QuantEngine.cagr(eps_3y_ago, eps_now, 3))
            except Exception:
                pass

        # ── FCF growth ────────────────────────────────────────────────────────
        fcf_now = ms.get("free_cash_flow")
        op_cf_row = get(raw.cash_flow, "Operating Cash Flow", "Total Cash From Operating Activities")
        capex_row = get(raw.cash_flow, "Capital Expenditure", "Capital Expenditures")

        op_cf_prior = prior(op_cf_row, 1)
        capex_prior = prior(capex_row, 1)
        if op_cf_prior is not None and capex_prior is not None and fcf_now is not None:
            fcf_prior = op_cf_prior + capex_prior
            if fcf_prior != 0:
                ms.set("fcf_growth_yoy", (fcf_now - fcf_prior) / abs(fcf_prior))

        # ── Gross margin trend (improving/stable/deteriorating) ───────────────
        gm_now = ms.get("gross_margin")
        gp_row = get(income, "Gross Profit")
        rev_prior_val = prior(rev_row, 1)
        gp_prior = prior(gp_row, 1)
        if gm_now and gp_prior and rev_prior_val and rev_prior_val > 0:
            gm_prior = gp_prior / rev_prior_val
            ms.set("gross_margin_trend", round(gm_now - gm_prior, 4))  # positive = improving

    # ── 4. Valuation Metrics ──────────────────────────────────────────────────

    def _compute_valuation_metrics(self, ms: MetricSet, raw: RawTickerData) -> None:
        info = raw.info
        prices = raw.price_history

        # Direct from info
        ms.set("pe_ratio", info.get("trailingPE"))
        ms.set("forward_pe", info.get("forwardPE"))
        ms.set("price_to_sales", info.get("priceToSalesTrailingTwelveMonths"))
        ms.set("price_to_book", info.get("priceToBook"))
        ms.set("enterprise_value", info.get("enterpriseValue"))

        ev = info.get("enterpriseValue")
        ebitda = info.get("ebitda")
        ebit_val = YFinanceFetcher.latest_value(
            YFinanceFetcher.get_statement_row(raw.income_stmt, "EBIT", "Operating Income")
        )

        if ev and ebitda and ebitda > 0:
            ms.set("ev_to_ebitda", ev / ebitda)
        if ev and ebit_val and ebit_val > 0:
            ms.set("ev_to_ebit", ev / ebit_val)

        # Historical P/E percentile (using price history + trailing EPS)
        if prices is not None and not prices.empty:
            eps_ttm = info.get("trailingEps")
            if eps_ttm and eps_ttm > 0:
                hist_pe = (prices["Close"] / eps_ttm).dropna()
                current_pe = ms.get("pe_ratio")
                if current_pe and len(hist_pe) >= 60:
                    ms.set(
                        "pe_percentile_5y",
                        QuantEngine.historical_percentile(current_pe, hist_pe.values)
                    )

            # Historical EV/EBITDA percentile (approximation using P/S as proxy)
            ps_current = ms.get("price_to_sales")
            rev_per_share = info.get("revenuePerShare")
            if rev_per_share and rev_per_share > 0 and ps_current:
                hist_ps = (prices["Close"] / rev_per_share).dropna()
                if len(hist_ps) >= 60:
                    ms.set(
                        "ps_percentile_2y",
                        QuantEngine.historical_percentile(ps_current, hist_ps.values)
                    )

        # FCF yield (already in fundamentals, but ensure it's here too)
        if ms.get("fcf_yield") is None:
            fcf = info.get("freeCashflow")
            mc = info.get("marketCap")
            if fcf and mc and mc > 0:
                ms.set("fcf_yield", fcf / mc)

    # ── 5. Composite Scores ───────────────────────────────────────────────────

    def _compute_composite_scores(self, ms: MetricSet, raw: RawTickerData) -> None:
        ms.set("piotroski_f_score", self._piotroski(ms, raw))
        ms.set("altman_z_score", self._altman_z(ms, raw))
        ms.set("earnings_quality_score", self._earnings_quality(ms))
        ms.set("overall_quality_score", self._overall_quality(ms))

    def _piotroski(self, ms: MetricSet, raw: RawTickerData) -> Optional[int]:
        """
        Piotroski F-Score (0–9). Higher = stronger financial position.
        3 dimensions: Profitability (4), Leverage/Liquidity (3), Operating efficiency (2).
        """
        score = 0
        signals = 0

        get = YFinanceFetcher.get_statement_row
        latest = YFinanceFetcher.latest_value
        prior = YFinanceFetcher.value_n_periods_ago

        # ── Profitability signals ─────────────────────────────────────────────
        # F1: ROA > 0
        roa = ms.get("roa")
        if roa is not None:
            if roa > 0:
                score += 1
            signals += 1

        # F2: Operating CF > 0
        op_cf = latest(get(raw.cash_flow, "Operating Cash Flow", "Total Cash From Operating Activities"))
        if op_cf is not None:
            if op_cf > 0:
                score += 1
            signals += 1

        # F3: ROA improving YoY
        net_income = latest(get(raw.income_stmt, "Net Income"))
        net_income_prior = prior(get(raw.income_stmt, "Net Income"), 1)
        total_assets = latest(get(raw.balance_sheet, "Total Assets"))
        total_assets_prior = prior(get(raw.balance_sheet, "Total Assets"), 1)
        if all(v is not None and v != 0 for v in [net_income, total_assets, net_income_prior, total_assets_prior]):
            roa_now = net_income / total_assets
            roa_prior = net_income_prior / total_assets_prior
            if roa_now > roa_prior:
                score += 1
            signals += 1

        # F4: Accruals (CF from ops / Total assets > ROA → cash earnings)
        if op_cf is not None and total_assets and total_assets > 0 and roa is not None:
            if (op_cf / total_assets) > roa:
                score += 1
            signals += 1

        # ── Leverage / Liquidity signals ──────────────────────────────────────
        # F5: Leverage (net_debt/ebitda) improved (lower is better)
        nde = ms.get("net_debt_to_ebitda")
        total_debt_now = latest(get(raw.balance_sheet, "Total Debt", "Long Term Debt"))
        total_debt_prior = prior(get(raw.balance_sheet, "Total Debt", "Long Term Debt"), 1)
        if total_debt_now is not None and total_debt_prior is not None:
            if total_debt_now <= total_debt_prior:
                score += 1
            signals += 1

        # F6: Current ratio improved
        cr_now = ms.get("current_ratio")
        curr_assets_prior = prior(get(raw.balance_sheet, "Current Assets", "Total Current Assets"), 1)
        curr_liab_prior = prior(get(raw.balance_sheet, "Current Liabilities", "Total Current Liabilities"), 1)
        if cr_now is not None and curr_assets_prior and curr_liab_prior and curr_liab_prior > 0:
            cr_prior = curr_assets_prior / curr_liab_prior
            if cr_now > cr_prior:
                score += 1
            signals += 1

        # F7: No new shares issued
        shares_now = raw.info.get("sharesOutstanding")
        # Approximate: if float shares ~= shares outstanding, minimal dilution
        shares_float = raw.info.get("floatShares")
        if shares_now and shares_float:
            if shares_float <= shares_now * 1.02:  # less than 2% dilution
                score += 1
            signals += 1

        # ── Operating Efficiency ──────────────────────────────────────────────
        # F8: Gross margin improved
        gm_trend = ms.get("gross_margin_trend")
        if gm_trend is not None:
            if gm_trend > 0:
                score += 1
            signals += 1

        # F9: Asset turnover improved (Revenue / Total Assets)
        rev_now = latest(get(raw.income_stmt, "Total Revenue"))
        rev_prior = prior(get(raw.income_stmt, "Total Revenue"), 1)
        if all(v and v > 0 for v in [rev_now, rev_prior, total_assets, total_assets_prior]):
            at_now = rev_now / total_assets
            at_prior = rev_prior / total_assets_prior
            if at_now > at_prior:
                score += 1
            signals += 1

        if signals < 5:
            return None  # Not enough data
        return score

    def _altman_z(self, ms: MetricSet, raw: RawTickerData) -> Optional[float]:
        """
        Altman Z-Score for public companies.
        Z > 2.99 → Safe zone, 1.81–2.99 → Grey zone, < 1.81 → Distress
        """
        try:
            get = YFinanceFetcher.get_statement_row
            latest = YFinanceFetcher.latest_value

            total_assets = latest(get(raw.balance_sheet, "Total Assets"))
            curr_assets = latest(get(raw.balance_sheet, "Current Assets", "Total Current Assets"))
            curr_liab = latest(get(raw.balance_sheet, "Current Liabilities", "Total Current Liabilities"))
            retained = latest(get(raw.balance_sheet, "Retained Earnings"))
            ebit = latest(get(raw.income_stmt, "EBIT", "Operating Income"))
            revenue = latest(get(raw.income_stmt, "Total Revenue"))
            total_debt = ms.get("total_debt") or 0
            market_cap = ms.get("market_cap")

            if not all([total_assets, curr_assets, curr_liab, ebit, revenue, market_cap]):
                return None
            if total_assets <= 0:
                return None

            wc = (curr_assets - curr_liab) / total_assets
            re = (retained or 0) / total_assets
            ebit_ta = ebit / total_assets
            mve_tl = market_cap / max(total_debt, 1)
            sales_ta = revenue / total_assets

            z = 1.2*wc + 1.4*re + 3.3*ebit_ta + 0.6*mve_tl + 1.0*sales_ta
            return round(z, 2)
        except Exception as e:
            logger.debug(f"[MetricsEngine] Altman Z failed: {e}")
            return None

    def _earnings_quality(self, ms: MetricSet) -> Optional[float]:
        """
        Earnings quality score (0–1).
        Measures how much earnings are backed by cash flow.
        High = cash-backed earnings (good), Low = accrual-heavy (risky).
        """
        cash_conv = ms.get("cash_conversion")
        fcf_margin = ms.get("fcf_margin")
        net_margin = ms.get("net_margin")

        scores = []
        if cash_conv is not None:
            # Clamp to 0-1 (>1 is great, <0 is bad)
            scores.append(max(0.0, min(1.0, cash_conv)))
        if fcf_margin is not None and net_margin is not None and net_margin != 0:
            ratio = fcf_margin / net_margin
            scores.append(max(0.0, min(1.0, ratio)))

        return round(sum(scores) / len(scores), 3) if scores else None

    def _overall_quality(self, ms: MetricSet) -> Optional[float]:
        """
        Composite quality score (0–10) blending all signal categories.
        Designed to be a quick single-number quality indicator for screeners.
        """
        sub_scores = []

        # Profitability (0-10)
        roic = ms.get("roic")
        if roic is not None:
            sub_scores.append(("profitability", min(10.0, max(0.0, roic * 40))))  # 25% ROIC → 10

        # FCF quality (0-10)
        fcf_yield = ms.get("fcf_yield")
        if fcf_yield is not None:
            sub_scores.append(("fcf_quality", min(10.0, max(0.0, fcf_yield * 200))))  # 5% yield → 10

        # Balance sheet (0-10) — lower net_debt/ebitda is better
        nde = ms.get("net_debt_to_ebitda")
        if nde is not None:
            bs_score = max(0.0, 10.0 - max(nde, 0) * 2)  # 0x → 10, 5x → 0
            sub_scores.append(("balance_sheet", bs_score))

        # Growth (0-10)
        rev_growth = ms.get("revenue_growth_yoy")
        if rev_growth is not None:
            sub_scores.append(("growth", min(10.0, max(0.0, rev_growth * 50))))  # 20% growth → 10

        # Earnings quality (0-10)
        eq = ms.get("earnings_quality_score")
        if eq is not None:
            sub_scores.append(("earnings_quality", eq * 10))

        if not sub_scores:
            return None

        total = sum(s for _, s in sub_scores) / len(sub_scores)
        return round(total, 2)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _period_return(close: pd.Series, delta_days: int) -> Optional[float]:
        if len(close) < delta_days + 1:
            return None
        try:
            return QuantEngine.simple_return(float(close.iloc[-(delta_days+1)]), float(close.iloc[-1]))
        except Exception:
            return None
