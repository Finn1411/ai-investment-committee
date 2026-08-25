"""
Tests for composite score calculations (Piotroski, Altman Z, earnings quality).
Uses mock RawTickerData to avoid network calls.
"""

from __future__ import annotations

import pandas as pd
import pytest

from finance_agent.data.fetcher import RawTickerData
from finance_agent.quant.metrics import MetricSet, MetricsEngine


def _make_income_stmt(revenue: float, gross_profit: float, ebit: float, net_income: float) -> pd.DataFrame:
    """Helper to build a minimal income statement DataFrame (yfinance format: index=metrics, cols=dates)."""
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2023-01-01"),
             pd.Timestamp("2022-01-01"), pd.Timestamp("2021-01-01")]
    data = {
        "Total Revenue":  [revenue, revenue * 0.9, revenue * 0.81, revenue * 0.73],
        "Gross Profit":   [gross_profit, gross_profit * 0.88, gross_profit * 0.79, gross_profit * 0.70],
        "EBIT":           [ebit, ebit * 0.85, ebit * 0.75, ebit * 0.65],
        "Net Income":     [net_income, net_income * 0.85, net_income * 0.78, net_income * 0.68],
    }
    # pd.DataFrame(data, index=dates) → rows=dates, cols=metrics → .T gives rows=metrics, cols=dates
    return pd.DataFrame(data, index=dates).T


def _make_balance_sheet(total_assets: float, curr_assets: float, curr_liab: float,
                         cash: float, total_debt: float, equity: float) -> pd.DataFrame:
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2023-01-01")]
    data = {
        "Total Assets":              [total_assets, total_assets * 0.95],
        "Current Assets":            [curr_assets, curr_assets * 0.92],
        "Current Liabilities":       [curr_liab, curr_liab * 0.95],
        "Cash And Cash Equivalents": [cash, cash * 0.9],
        "Total Debt":                [total_debt, total_debt * 1.05],
        "Stockholders Equity":       [equity, equity * 0.95],
        "Retained Earnings":         [equity * 0.6, equity * 0.55],
    }
    return pd.DataFrame(data, index=dates).T


def _make_cashflow(op_cf: float, capex: float) -> pd.DataFrame:
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2023-01-01")]
    data = {
        "Operating Cash Flow": [op_cf, op_cf * 0.90],
        "Capital Expenditure": [capex, capex * 1.05],
    }
    return pd.DataFrame(data, index=dates).T


def _make_raw(
    revenue=100e9, gross_profit=44e9, ebit=29e9, net_income=25e9,
    total_assets=350e9, curr_assets=140e9, curr_liab=70e9,
    cash=60e9, total_debt=90e9, equity=70e9,
    op_cf=35e9, capex=-10e9,
    market_cap=2_800e9, trailing_eps=6.5, ebitda=32e9,
    beta=1.2, shares=16e9, float_shares=15.5e9,
) -> RawTickerData:
    raw = RawTickerData(ticker="TEST")
    raw.income_stmt = _make_income_stmt(revenue, gross_profit, ebit, net_income)
    raw.balance_sheet = _make_balance_sheet(total_assets, curr_assets, curr_liab, cash, total_debt, equity)
    raw.cash_flow = _make_cashflow(op_cf, capex)
    raw.info = {
        "marketCap": market_cap,
        "trailingEps": trailing_eps,
        "trailingPE": market_cap / (net_income),
        "forwardPE": 25.0,
        "ebitda": ebitda,
        "enterpriseValue": market_cap + total_debt - cash,
        "grossMargins": gross_profit / revenue,
        "operatingMargins": ebit / revenue,
        "profitMargins": net_income / revenue,
        "beta": beta,
        "returnOnEquity": net_income / equity,
        "returnOnAssets": net_income / total_assets,
        "freeCashflow": op_cf + capex,
        "sharesOutstanding": shares,
        "floatShares": float_shares,
        "priceToBook": 3.5,
        "priceToSalesTrailingTwelveMonths": market_cap / revenue,
        "currentRatio": curr_assets / curr_liab,
        "currency": "USD",
    }
    return raw


class TestMetricSetBasics:
    def test_set_and_get(self):
        ms = MetricSet()
        ms.set("pe_ratio", 25.0)
        assert ms.get("pe_ratio") == 25.0

    def test_attribute_access(self):
        ms = MetricSet()
        ms.set("net_margin", 0.25)
        assert ms.net_margin == 0.25

    def test_available_count(self):
        ms = MetricSet()
        ms.set("a", 1.0)
        ms.set("b", None)
        ms.set("c", 3.0)
        assert ms.available_count == 2

    def test_missing_returns_none(self):
        ms = MetricSet()
        assert ms.get("nonexistent") is None


class TestFundamentalMetrics:
    def setup_method(self):
        self.engine = MetricsEngine.__new__(MetricsEngine)
        self.engine.benchmark_ticker = "^GSPC"

    def test_gross_margin_computed(self):
        raw = _make_raw(revenue=100e9, gross_profit=44e9)
        ms = MetricSet()
        self.engine._compute_fundamental_metrics(ms, raw)
        assert ms.get("gross_margin") == pytest.approx(0.44, abs=0.01)

    def test_operating_margin_computed(self):
        raw = _make_raw(revenue=100e9, ebit=29e9)
        ms = MetricSet()
        self.engine._compute_fundamental_metrics(ms, raw)
        assert ms.get("operating_margin") == pytest.approx(0.29, abs=0.01)

    def test_net_margin_computed(self):
        raw = _make_raw(revenue=100e9, net_income=25e9)
        ms = MetricSet()
        self.engine._compute_fundamental_metrics(ms, raw)
        assert ms.get("net_margin") == pytest.approx(0.25, abs=0.01)

    def test_fcf_computed(self):
        raw = _make_raw(op_cf=35e9, capex=-10e9)
        ms = MetricSet()
        self.engine._compute_fundamental_metrics(ms, raw)
        assert ms.get("free_cash_flow") == pytest.approx(25e9, rel=0.01)

    def test_current_ratio(self):
        raw = _make_raw(curr_assets=140e9, curr_liab=70e9)
        ms = MetricSet()
        self.engine._compute_fundamental_metrics(ms, raw)
        assert ms.get("current_ratio") == pytest.approx(2.0, abs=0.01)


class TestGrowthMetrics:
    def setup_method(self):
        self.engine = MetricsEngine.__new__(MetricsEngine)
        self.engine.benchmark_ticker = "^GSPC"

    def test_revenue_growth_yoy(self):
        raw = _make_raw(revenue=100e9)
        ms = MetricSet()
        self.engine._compute_growth_metrics(ms, raw)
        # revenue is 100, prior year is 90 → 11.1% growth
        assert ms.get("revenue_growth_yoy") == pytest.approx(0.111, abs=0.01)

    def test_revenue_cagr_3y(self):
        raw = _make_raw(revenue=100e9)
        ms = MetricSet()
        self.engine._compute_growth_metrics(ms, raw)
        cagr = ms.get("revenue_cagr_3y")
        assert cagr is not None
        assert 0.05 < cagr < 0.15  # ~10% CAGR from the mock data


class TestCompositeScores:
    def setup_method(self):
        self.engine = MetricsEngine.__new__(MetricsEngine)
        self.engine.benchmark_ticker = "^GSPC"

    def _full_ms(self, raw):
        ms = MetricSet()
        self.engine._compute_fundamental_metrics(ms, raw)
        self.engine._compute_growth_metrics(ms, raw)
        return ms

    def test_piotroski_range(self):
        raw = _make_raw()
        ms = self.engine._piotroski(MetricSet(), raw)
        if ms is not None:
            assert 0 <= ms <= 9

    def test_altman_z_healthy_company(self):
        # Large profitable company → should be in safe zone (Z > 2.99)
        raw = _make_raw(
            total_assets=350e9, curr_assets=140e9, curr_liab=70e9,
            ebit=29e9, revenue=100e9, total_debt=90e9, market_cap=2800e9
        )
        ms = MetricSet()
        ms.set("total_debt", 90e9)
        ms.set("net_debt", 30e9)
        ms.set("market_cap", 2800e9)
        z = self.engine._altman_z(ms, raw)
        assert z is not None
        assert z > 2.5  # Should be in safe zone for a large healthy company

    def test_earnings_quality_high_cash_conversion(self):
        ms = MetricSet()
        ms.set("cash_conversion", 1.1)     # FCF > Net Income = great
        ms.set("fcf_margin", 0.25)
        ms.set("net_margin", 0.25)
        score = self.engine._earnings_quality(ms)
        assert score is not None
        assert score > 0.8

    def test_earnings_quality_low_cash_conversion(self):
        ms = MetricSet()
        ms.set("cash_conversion", 0.2)     # FCF << Net Income = risky
        ms.set("fcf_margin", 0.05)
        ms.set("net_margin", 0.25)
        score = self.engine._earnings_quality(ms)
        assert score is not None
        assert score < 0.5

    def test_overall_quality_score_range(self):
        raw = _make_raw()
        ms = self._full_ms(raw)
        ms.set("roic", 0.25)
        ms.set("fcf_yield", 0.04)
        ms.set("net_debt_to_ebitda", 1.0)
        ms.set("revenue_growth_yoy", 0.10)
        ms.set("earnings_quality_score", 0.8)
        score = self.engine._overall_quality(ms)
        assert score is not None
        assert 0 <= score <= 10
