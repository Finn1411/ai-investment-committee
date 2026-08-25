"""
Smoke tests for the Quant Engine.
Run with: pytest tests/ -v
"""

import math

import numpy as np
import pytest

from finance_agent.quant.engine import QuantEngine


class TestReturns:
    def test_simple_return(self):
        assert QuantEngine.simple_return(100, 110) == pytest.approx(0.10)

    def test_simple_return_negative(self):
        assert QuantEngine.simple_return(100, 80) == pytest.approx(-0.20)

    def test_log_return(self):
        r = QuantEngine.log_return(100, 110)
        assert abs(r - math.log(1.1)) < 1e-9

    def test_cagr(self):
        # 100 → 121 in 2 years = 10% CAGR
        assert QuantEngine.cagr(100, 121, 2) == pytest.approx(0.10, abs=1e-6)

    def test_cagr_from_series(self):
        revenues = [100, 110, 121, 133.1]
        result = QuantEngine.cagr_from_series(revenues)
        assert result == pytest.approx(0.10, abs=1e-4)


class TestRisk:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.daily_returns = rng.normal(0.0003, 0.012, 252)

    def test_volatility_positive(self):
        vol = QuantEngine.annualised_volatility(self.daily_returns)
        assert vol > 0

    def test_max_drawdown_negative(self):
        prices = np.array([100, 110, 95, 120, 80, 130])
        dd = QuantEngine.maximum_drawdown(prices)
        assert dd < 0

    def test_max_drawdown_no_drawdown(self):
        prices = np.array([100, 110, 120, 130])
        dd = QuantEngine.maximum_drawdown(prices)
        assert dd == pytest.approx(0.0)

    def test_beta_market(self):
        # Asset identical to benchmark → beta = 1
        r = np.random.default_rng(0).normal(0, 0.01, 252)
        b = QuantEngine.beta(r, r)
        assert b == pytest.approx(1.0, abs=1e-9)

    def test_sharpe_ratio(self):
        sr = QuantEngine.sharpe_ratio(self.daily_returns)
        assert isinstance(sr, float)

    def test_sortino_ratio(self):
        so = QuantEngine.sortino_ratio(self.daily_returns)
        assert isinstance(so, float)

    def test_var(self):
        var = QuantEngine.value_at_risk(self.daily_returns)
        assert var < 0  # 5% VaR should be negative


class TestValuation:
    def test_historical_percentile_median(self):
        hist = list(range(1, 101))
        assert QuantEngine.historical_percentile(50, hist) == pytest.approx(50.0)

    def test_historical_percentile_max(self):
        hist = list(range(1, 101))
        assert QuantEngine.historical_percentile(100, hist) == pytest.approx(100.0)

    def test_fcf_yield(self):
        assert QuantEngine.fcf_yield(10_000_000, 200_000_000) == pytest.approx(0.05)

    def test_ev(self):
        ev = QuantEngine.ev(market_cap=100, total_debt=30, cash=10)
        assert ev == 120

    def test_ev_to_ebitda(self):
        assert QuantEngine.ev_to_ebitda(120, 10) == pytest.approx(12.0)

    def test_ev_to_ebitda_negative_ebitda(self):
        assert QuantEngine.ev_to_ebitda(120, -5) is None


class TestDCF:
    def test_dcf_positive(self):
        val = QuantEngine.simple_dcf(
            fcf_base=100, growth_rate=0.10, terminal_growth=0.03,
            discount_rate=0.10, forecast_years=10
        )
        assert val > 0

    def test_reverse_dcf(self):
        # Build a DCF value then reverse it
        true_growth = 0.15
        ev = QuantEngine.simple_dcf(
            fcf_base=100, growth_rate=true_growth, terminal_growth=0.03,
            discount_rate=0.10, forecast_years=10
        )
        implied = QuantEngine.reverse_dcf_growth(
            current_ev=ev, fcf_base=100, terminal_growth=0.03,
            discount_rate=0.10, forecast_years=10
        )
        assert implied == pytest.approx(true_growth, abs=0.005)


class TestMonteCarlo:
    def test_mc_shape(self):
        results = QuantEngine.monte_carlo_returns(0.10, 0.20, n_simulations=1000)
        assert results.shape == (1000,)

    def test_mc_probabilities_sum(self):
        mc = QuantEngine.monte_carlo_returns(0.08, 0.18, n_simulations=5000)
        probs = QuantEngine.scenario_probabilities_from_mc(mc)
        total = probs["bear_prob"] + probs["base_prob"] + probs["bull_prob"]
        assert total == pytest.approx(1.0, abs=0.001)
