"""
Tests for the Scenario Builder.
No network calls — uses MetricSet constructed from known values.
"""

from __future__ import annotations

import math
import pytest

from finance_agent.models.schemas import Horizon
from finance_agent.quant.metrics import MetricSet
from finance_agent.quant.scenarios import ScenarioBuilder, ScenarioInputs, build_scenarios_from_metrics


def _good_company_inputs() -> ScenarioInputs:
    return ScenarioInputs(
        ticker="TEST",
        horizon=Horizon.TWELVE_MONTHS,
        current_price=150.0,
        beta=1.1,
        volatility_annual=0.22,
        revenue_growth_yoy=0.12,
        eps_growth_yoy=0.15,
        fcf_growth_yoy=0.18,
        roic=0.25,
        net_debt_to_ebitda=0.5,
        piotroski_f_score=7,
        pe_ratio=25.0,
        forward_pe=21.0,
        pe_percentile_5y=60.0,
        fcf_yield=0.04,
        n_simulations=5000,
        seed=42,
    )


class TestScenarioBuilder:
    def test_probabilities_sum_to_one(self):
        inputs = _good_company_inputs()
        result = ScenarioBuilder().build(inputs)
        total = (result.model.bear.probability +
                 result.model.base.probability +
                 result.model.bull.probability)
        assert total == pytest.approx(1.0, abs=0.001)

    def test_bear_return_less_than_bull(self):
        result = ScenarioBuilder().build(_good_company_inputs())
        assert result.model.bear.expected_return < result.model.bull.expected_return

    def test_expected_value_in_reasonable_range(self):
        result = ScenarioBuilder().build(_good_company_inputs())
        ev = result.model.expected_value
        assert -0.50 < ev < 0.80

    def test_all_scenarios_have_narratives(self):
        result = ScenarioBuilder().build(_good_company_inputs())
        assert result.model.bear.narrative
        assert result.model.base.narrative
        assert result.model.bull.narrative

    def test_high_leverage_in_bear_narrative(self):
        inputs = _good_company_inputs()
        inputs.net_debt_to_ebitda = 5.0  # High leverage
        result = ScenarioBuilder().build(inputs)
        assert "leverage" in result.model.bear.narrative.lower()

    def test_override_returns_respected(self):
        inputs = _good_company_inputs()
        inputs.bear_return_override = -0.40
        inputs.base_return_override = 0.12
        inputs.bull_return_override = 0.55
        result = ScenarioBuilder().build(inputs)
        assert result.model.bear.expected_return == pytest.approx(-0.40, abs=0.001)
        assert result.model.base.expected_return == pytest.approx(0.12, abs=0.001)
        assert result.model.bull.expected_return == pytest.approx(0.55, abs=0.001)

    def test_three_month_horizon(self):
        inputs = _good_company_inputs()
        inputs.horizon = Horizon.THREE_MONTHS
        result = ScenarioBuilder().build(inputs)
        # 3M range should be tighter than 12M
        range_3m = result.model.bull.expected_return - result.model.bear.expected_return
        inputs.horizon = Horizon.TWELVE_MONTHS
        result_12m = ScenarioBuilder().build(inputs)
        range_12m = result_12m.model.bull.expected_return - result_12m.model.bear.expected_return
        assert range_3m < range_12m

    def test_build_from_metricset(self):
        ms = MetricSet()
        ms.set("current_price", 150.0)
        ms.set("beta", 1.1)
        ms.set("volatility_90d", 0.22)
        ms.set("revenue_growth_yoy", 0.12)
        ms.set("fcf_yield", 0.04)
        ms.set("roic", 0.20)
        ms.set("pe_percentile_5y", 60.0)
        ms.set("market_cap", 2_500e9)

        result = build_scenarios_from_metrics("TEST", ms, Horizon.TWELVE_MONTHS)
        total = (result.model.bear.probability +
                 result.model.base.probability +
                 result.model.bull.probability)
        assert total == pytest.approx(1.0, abs=0.001)
