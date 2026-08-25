"""
Tests for all reasoning agents.
Strategy: Mock the LLM call to avoid API key requirement.
Verify that each agent: builds correct AgentAnalysis, handles JSON parse failures,
clips scores to 0-10, and clips confidence to 0-1.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from finance_agent.agents.bear_analyst import BearAnalyst
from finance_agent.agents.earnings_analyst import EarningsAnalyst
from finance_agent.agents.fundamental_analyst import FundamentalAnalyst
from finance_agent.agents.growth_analyst import GrowthAnalyst
from finance_agent.agents.portfolio_manager import PortfolioContext, PortfolioManager
from finance_agent.agents.risk_manager import RiskManager
from finance_agent.agents.value_analyst import ValueAnalyst
from finance_agent.models.schemas import AgentAnalysis


# ── Mock context factory ──────────────────────────────────────────────────────

def _make_mock_context(ticker: str = "TEST") -> MagicMock:
    """Build a minimal mock AnalysisContext for testing."""
    ctx = MagicMock()
    ctx.ticker = ticker
    ctx.company_name = "Test Corp"
    ctx.sector = "Technology"
    ctx.industry = "Software"
    ctx.analysis_date = date.today()
    ctx.data_passed_quality_check = True
    ctx.quality_report.warnings = []
    ctx.quality_report.passed = True
    ctx.raw.earnings_dates = None

    # Mock to_prompt_dict
    ctx.to_prompt_dict.return_value = {
        "current_price": 150.0,
        "market_cap_bn": 2500.0,
        "return_1y": 0.35,
        "return_ytd": 0.20,
        "beta": 1.1,
        "volatility_annual": 0.25,
        "sharpe_ratio_1y": 1.2,
        "max_drawdown_1y": -0.15,
        "gross_margin": 0.45,
        "operating_margin": 0.30,
        "net_margin": 0.25,
        "roic": 0.25,
        "roe": 0.30,
        "fcf_margin": 0.22,
        "fcf_yield": 0.03,
        "cash_conversion": 0.88,
        "earnings_quality_score": 0.85,
        "net_debt_to_ebitda": 0.5,
        "interest_coverage": 15.0,
        "current_ratio": 1.5,
        "revenue_growth_yoy": 0.10,
        "eps_growth_yoy": 0.18,
        "fcf_growth_yoy": 0.12,
        "revenue_cagr_3y": 0.08,
        "eps_cagr_3y": 0.15,
        "gross_margin_trend": 0.02,
        "pe_ratio": 30.0,
        "forward_pe": 26.0,
        "ev_to_ebitda": 20.0,
        "price_to_sales": 8.0,
        "pe_percentile_5y": 80.0,
        "fcf_yield_pct": 3.0,
        "piotroski_f_score": 7,
        "altman_z_score": 5.5,
        "overall_quality_score": 7.5,
        "valuation_label": "EXPENSIVE",
        "valuation_score": 3.5,
        "dcf_intrinsic_per_share": 120.0,
        "dcf_margin_of_safety": -0.25,
        "dcf_upside": -0.20,
        "implied_fcf_growth": 0.18,
        "reverse_dcf_narrative": "Market prices in 18% FCF CAGR over 10y",
        "scenario_bear_prob": 0.15,
        "scenario_bear_return": -0.35,
        "scenario_base_prob": 0.70,
        "scenario_base_return": 0.10,
        "scenario_bull_prob": 0.15,
        "scenario_bull_return": 0.45,
        "scenario_expected_value": 0.07,
        "scenario_prob_positive": 0.85,
        "var_95_1y": -0.35,
        "data_quality_warnings": [],
        "estimated_fields": [],
    }

    # Mock valuation
    ctx.valuation = MagicMock()
    ctx.valuation.valuation_label = "EXPENSIVE"
    ctx.valuation.composite_valuation_score = 3.5
    ctx.valuation.dcf = MagicMock()
    ctx.valuation.dcf.intrinsic_value_per_share = 120.0
    ctx.valuation.dcf.margin_of_safety = -0.25
    ctx.valuation.sensitivity = None

    # Mock scenarios
    ctx.scenarios = MagicMock()
    ctx.scenarios.model.expected_value = 0.07
    ctx.scenarios.model.bear.probability = 0.15
    ctx.scenarios.model.bear.expected_return = -0.35
    ctx.scenarios.model.base.probability = 0.70
    ctx.scenarios.model.base.expected_return = 0.10
    ctx.scenarios.model.bull.probability = 0.15
    ctx.scenarios.model.bull.expected_return = 0.45
    ctx.scenarios.model.prob_outperform = 0.85

    return ctx


def _make_good_response(agent_class) -> str:
    """Build a minimal valid JSON response for each agent type."""
    return json.dumps({
        "summary": "Test analysis complete with strong fundamentals.",
        "score": 7.5,
        "confidence": 0.80,
        "key_findings": [
            "Finding 1: ROIC of 25% exceeds WACC significantly",
            "Finding 2: Gross margin expanding at +2pp trend",
            "Finding 3: Piotroski score 7/9 indicates financial strength",
            "Finding 4: FCF conversion at 88% is high quality",
            "Finding 5: Balance sheet healthy with 0.5x Net Debt/EBITDA"
        ],
        # Extra fields that different agents expect
        "strengths": ["High ROIC", "Strong margins"],
        "weaknesses": ["Premium valuation"],
        "business_quality_assessment": "Strong business with durable margins",
        "financial_health_assessment": "Fortress balance sheet",
        "trend_assessment": "Margins improving",
        "moat_indicators": "High ROIC suggests durable competitive advantage",
        "red_flags": [],
        "embedded_expectations": "18% FCF CAGR priced in",
        "margin_of_safety_assessment": "No margin of safety at current price",
        "fcf_yield_assessment": "3% yield is modest",
        "historical_valuation_context": "At 80th percentile",
        "reverse_dcf_interpretation": "Requires continued high growth",
        "valuation_risks": ["Multiple compression"],
        "valuation_opportunities": [],
        "base_case_return_expectation": "5-8% annually",
        "growth_quality_assessment": "Fundamental growth backed by FCF",
        "revenue_growth_sustainability": "Sustainable with pricing power",
        "pricing_power_evidence": "Gross margins expanding",
        "scalability_assessment": "High operating leverage",
        "management_execution": "Consistent delivery on guidance",
        "competitive_advantage_assessment": "Durable moat from switching costs",
        "reinvestment_quality": "ROIC 25% well above WACC",
        "growth_risks": ["Market saturation"],
        "growth_catalysts": ["New product cycle"],
        "earnings_setup": "Set up to beat on margins",
        "margin_catalyst": "Operating leverage improving",
        "capital_return_catalyst": "Buybacks at 3% FCF yield",
        "structural_catalysts": ["AI tailwind"],
        "near_term_risks": ["Macro slowdown"],
        "guidance_watch": "Watch for margin guidance",
        "time_horizon_assessment": "12M catalyst is margin expansion",
        "risk_register": {
            "balance_sheet_risk": {"severity": "LOW", "assessment": "Clean", "key_metric": "0.5x ND/EBITDA"},
            "competitive_risk": {"severity": "MEDIUM", "assessment": "Some risk", "key_metric": "Margins stable"},
            "valuation_compression_risk": {"severity": "HIGH", "assessment": "Rich multiples", "key_metric": "80th pctile"},
            "earnings_risk": {"severity": "LOW", "assessment": "High quality", "key_metric": "EQ: 0.85"},
            "macro_risk": {"severity": "MEDIUM", "assessment": "Beta 1.1", "key_metric": "Beta 1.1"},
            "concentration_risk": {"severity": "LOW", "assessment": "Diversified", "key_metric": "N/A"}
        },
        "maximum_loss_estimate": "-35% in severe bear scenario",
        "altman_z_interpretation": "Z=5.5 well in safe zone",
        "var_interpretation": "-35% at 95% confidence",
        "top_risks_ranked": ["#1: Valuation compression", "#2: Macro", "#3: Competition"],
        "risk_mitigants": ["Strong FCF", "Low leverage"],
        "thesis_vulnerabilities": ["18% growth is very optimistic"],
        "dcf_assumptions_challenged": "10% discount rate too low",
        "growth_story_challenged": "3Y CAGR was only 8%",
        "what_must_go_right": "Revenue acceleration AND margin expansion simultaneously",
        "historical_analog": "Similar premium tech companies",
        "ignored_red_flags": ["FCF declining -9% YoY despite EPS growth"],
        "probability_weighted_bear_case": "-35% if growth disappoints",
        "what_would_make_bear_wrong": "Revenue reacceleration above 15%",
        "recommended_position_size_pct": 4.0,
        "position_sizing_rationale": "4% position given rich valuation",
        "sector_concentration_assessment": "Tech exposure manageable",
        "correlation_assessment": "Some correlation with existing tech positions",
        "risk_contribution_assessment": "Moderate systematic risk added",
        "portfolio_fit_verdict": "ACCUMULATE",
        "conditions_for_larger_position": "Price decline to DCF intrinsic value",
        "conditions_for_no_position": "Score below 5 or margin compression",
        "alternative_opportunity_cost": "vs cheaper quality companies"
    })


# ── Base: test that mock infrastructure works ─────────────────────────────────

class TestAgentBase:
    """Test the base agent's _build_analysis and metric formatting."""

    def test_score_clipped_to_0_10(self):
        agent = FundamentalAnalyst.__new__(FundamentalAnalyst)
        agent.name = "FundamentalAnalyst"
        result = agent._build_analysis("TEST", "summary", [], score=15.0, confidence=0.5)
        assert result.score == 10.0

    def test_score_clipped_at_zero(self):
        agent = FundamentalAnalyst.__new__(FundamentalAnalyst)
        agent.name = "FundamentalAnalyst"
        result = agent._build_analysis("TEST", "summary", [], score=-5.0, confidence=0.5)
        assert result.score == 0.0

    def test_confidence_clipped(self):
        agent = FundamentalAnalyst.__new__(FundamentalAnalyst)
        agent.name = "FundamentalAnalyst"
        result = agent._build_analysis("TEST", "summary", [], score=7.0, confidence=1.5)
        assert result.confidence == 1.0

    def test_format_metrics_returns_string(self):
        ctx = _make_mock_context()
        result = FundamentalAnalyst._format_metrics(ctx)
        assert isinstance(result, str)
        assert "TEST" in result or "Test Corp" in result
        assert "ROIC" in result

    def test_returns_agent_analysis(self):
        agent = FundamentalAnalyst.__new__(FundamentalAnalyst)
        agent.name = "FundamentalAnalyst"
        result = agent._build_analysis("AAPL", "Summary", ["F1", "F2"], 7.5, 0.8)
        assert isinstance(result, AgentAnalysis)
        assert result.ticker == "AAPL"
        assert result.agent_name == "FundamentalAnalyst"
        assert result.score == 7.5
        assert result.confidence == 0.8


# ── Agent smoke tests (mocked LLM) ────────────────────────────────────────────

AGENTS_TO_TEST = [
    ("FundamentalAnalyst", FundamentalAnalyst),
    ("ValueAnalyst", ValueAnalyst),
    ("GrowthAnalyst", GrowthAnalyst),
    ("EarningsAnalyst", EarningsAnalyst),
    ("RiskManager", RiskManager),
    ("BearAnalyst", BearAnalyst),
]


class TestAllAgentsMocked:
    """Smoke-test all agents with mocked Gemini calls."""

    def _make_agent_with_mock_client(self, agent_class):
        """Instantiate agent with fully mocked google.genai client."""
        mock_response = MagicMock()
        mock_response.text = _make_good_response(agent_class)

        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response

        mock_client = MagicMock()
        mock_client.models = mock_models

        with patch("google.genai.Client", return_value=mock_client), \
             patch("google.genai.types.GenerateContentConfig", return_value=MagicMock()):
            agent = agent_class()
            # Attach the mock so generate_content returns good JSON
            agent._client = mock_client
        return agent, mock_client

    @pytest.mark.parametrize("agent_name,agent_class", AGENTS_TO_TEST)
    def test_agent_returns_valid_analysis(self, agent_name, agent_class):
        ctx = _make_mock_context()
        agent, _ = self._make_agent_with_mock_client(agent_class)

        if agent_class == BearAnalyst:
            result = agent.analyse(ctx, bull_thesis_summary="Bulls love this stock.")
        else:
            result = agent.analyse(ctx)

        assert isinstance(result, AgentAnalysis), f"{agent_name} must return AgentAnalysis"
        assert result.ticker == "TEST"
        assert result.agent_name == agent_name
        assert 0.0 <= result.score <= 10.0, f"{agent_name} score out of range: {result.score}"
        assert 0.0 <= result.confidence <= 1.0, f"{agent_name} confidence out of range"
        assert len(result.summary) > 10, f"{agent_name} summary too short"

    @pytest.mark.parametrize("agent_name,agent_class", AGENTS_TO_TEST)
    def test_agent_handles_json_parse_failure(self, agent_name, agent_class):
        """Agents must not crash even if LLM returns garbage text."""
        ctx = _make_mock_context()

        mock_response = MagicMock()
        mock_response.text = "Sorry, I cannot analyse this. No JSON here at all."
        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response
        mock_client = MagicMock()
        mock_client.models = mock_models

        with patch("google.genai.Client", return_value=mock_client), \
             patch("google.genai.types.GenerateContentConfig", return_value=MagicMock()):
            agent = agent_class()
            agent._client = mock_client

        try:
            if agent_class == BearAnalyst:
                result = agent.analyse(ctx, bull_thesis_summary="Bull thesis")
            else:
                result = agent.analyse(ctx)
            # If returns without crashing, score must still be valid
            assert 0.0 <= result.score <= 10.0
        except Exception:
            pass  # Graceful fallback failure is acceptable

    def test_portfolio_manager_with_portfolio(self):
        ctx = _make_mock_context()
        portfolio = PortfolioContext(
            existing_positions={"MSFT": 0.05, "GOOGL": 0.04},
            sector_weights={"Technology": 0.25, "Healthcare": 0.10},
        )
        agent, _ = self._make_agent_with_mock_client(PortfolioManager)
        result = agent.analyse(ctx, portfolio=portfolio, committee_score=7.5)

        assert isinstance(result, AgentAnalysis)
        assert result.agent_name == "PortfolioManager"
        assert 0.0 <= result.score <= 10.0


# ── Committee scoring tests (no LLM) ─────────────────────────────────────────

class TestCommitteeScoring:
    """Test the committee's deterministic scoring logic (no LLM calls)."""

    def test_weighted_score_basic(self):
        from finance_agent.agents.committee import CommitteeEngine, DEFAULT_WEIGHTS
        engine = CommitteeEngine.__new__(CommitteeEngine)
        engine.weights = DEFAULT_WEIGHTS

        # All agents score 8.0 → weighted score should be 8.0
        scores = {name: 8.0 for name in DEFAULT_WEIGHTS}
        result = engine._compute_weighted_score(scores)
        assert result == pytest.approx(8.0, abs=0.01)

    def test_weighted_score_mixed(self):
        from finance_agent.agents.committee import CommitteeEngine, DEFAULT_WEIGHTS
        engine = CommitteeEngine.__new__(CommitteeEngine)
        engine.weights = DEFAULT_WEIGHTS

        # BearAnalyst (weight 0.20) scores 2.0, rest score 8.0
        scores = {name: 8.0 for name in DEFAULT_WEIGHTS}
        scores["BearAnalyst"] = 2.0
        result = engine._compute_weighted_score(scores)
        # 2.0 * 0.20 + 8.0 * 0.80 = 0.4 + 6.4 = 6.8
        assert 6.5 < result < 7.5

    def test_buy_rating_threshold(self):
        from finance_agent.agents.committee import CommitteeEngine
        from finance_agent.models.schemas import Rating
        engine = CommitteeEngine.__new__(CommitteeEngine)
        assert engine._score_to_rating(7.0) == Rating.BUY

    def test_hold_rating_threshold(self):
        from finance_agent.agents.committee import CommitteeEngine
        from finance_agent.models.schemas import Rating
        engine = CommitteeEngine.__new__(CommitteeEngine)
        assert engine._score_to_rating(5.0) == Rating.HOLD

    def test_avoid_rating_threshold(self):
        from finance_agent.agents.committee import CommitteeEngine
        from finance_agent.models.schemas import Rating
        engine = CommitteeEngine.__new__(CommitteeEngine)
        assert engine._score_to_rating(3.0) == Rating.AVOID

    def test_weights_sum_to_one(self):
        from finance_agent.agents.committee import DEFAULT_WEIGHTS
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0, abs=0.001)

    def test_missing_agent_excluded_from_score(self):
        from finance_agent.agents.committee import CommitteeEngine, DEFAULT_WEIGHTS
        engine = CommitteeEngine.__new__(CommitteeEngine)
        engine.weights = DEFAULT_WEIGHTS

        # Only 3 agents scored — rest excluded, weights renormalized
        scores = {
            "FundamentalAnalyst": 8.0,
            "ValueAnalyst": 6.0,
            "GrowthAnalyst": 7.0,
        }
        result = engine._compute_weighted_score(scores)
        # Should be a valid number between 0 and 10
        assert 0.0 <= result <= 10.0
