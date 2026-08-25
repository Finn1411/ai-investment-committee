"""
Week 4 integration tests.

Tests the full pipeline from CommitteeEngine to CommitteeVerdict
including disagreement detection, narrative extraction, and JSON/Markdown export.
All LLM calls and yfinance calls are mocked.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from finance_agent.agents.committee import CommitteeEngine, CommitteeResult, DEFAULT_WEIGHTS
from finance_agent.models.schemas import (
    CommitteeVerdict, Horizon, Rating, Scenario, ScenarioModel,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scenario_model(ticker: str = "TEST") -> ScenarioModel:
    return ScenarioModel(
        ticker=ticker,
        horizon=Horizon.TWELVE_MONTHS,
        analysis_date=date.today(),
        bear=Scenario(label="Bear", probability=0.15, expected_return=-0.35,
                      narrative="Competition intensifies"),
        base=Scenario(label="Base", probability=0.70, expected_return=0.08,
                      narrative="Steady growth continues"),
        bull=Scenario(label="Bull", probability=0.15, expected_return=0.45,
                      narrative="Market expansion accelerates"),
    )


def _make_mock_context(ticker: str = "TEST") -> MagicMock:
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
    ctx.scenarios.model = _make_scenario_model(ticker)
    ctx.valuation.valuation_label = "FAIR"
    ctx.valuation.sensitivity = None

    ctx.to_prompt_dict.return_value = {
        "current_price": 150.0,
        "market_cap_bn": 500.0,
        "beta": 1.1,
        "gross_margin": 0.45,
        "operating_margin": 0.28,
        "net_margin": 0.22,
        "roic": 0.20,
        "roe": 0.25,
        "fcf_margin": 0.18,
        "fcf_yield": 0.03,
        "cash_conversion": 0.85,
        "earnings_quality_score": 0.82,
        "net_debt_to_ebitda": 0.8,
        "interest_coverage": 10.0,
        "current_ratio": 1.5,
        "revenue_growth_yoy": 0.12,
        "eps_growth_yoy": 0.18,
        "fcf_growth_yoy": 0.10,
        "revenue_cagr_3y": 0.10,
        "eps_cagr_3y": 0.15,
        "gross_margin_trend": 0.01,
        "pe_ratio": 25.0,
        "forward_pe": 22.0,
        "ev_to_ebitda": 18.0,
        "price_to_sales": 6.0,
        "pe_percentile_5y": 70.0,
        "piotroski_f_score": 7,
        "altman_z_score": 4.5,
        "overall_quality_score": 7.0,
        "valuation_label": "FAIR",
        "valuation_score": 5.0,
        "dcf_intrinsic_per_share": 145.0,
        "dcf_margin_of_safety": -0.03,
        "implied_fcf_growth": 0.12,
        "reverse_dcf_narrative": "Market prices in 12% FCF CAGR",
        "scenario_bear_prob": 0.15,
        "scenario_bear_return": -0.35,
        "scenario_base_prob": 0.70,
        "scenario_base_return": 0.08,
        "scenario_bull_prob": 0.15,
        "scenario_bull_return": 0.45,
        "scenario_expected_value": 0.065,
        "var_95_1y": -0.30,
        "return_1y": 0.25,
        "return_ytd": 0.15,
        "volatility_annual": 0.22,
        "sharpe_ratio_1y": 1.0,
        "max_drawdown_1y": -0.12,
    }
    return ctx


def _make_agent_response(score: float, summary_prefix: str = "") -> dict:
    summary = f"{summary_prefix}Test analysis complete. Strong fundamentals support this view."
    return {
        "summary": summary,
        "score": score,
        "confidence": 0.80,
        "key_findings": [
            f"Finding 1: ROIC of 20% exceeds WACC (score context: {score:.1f})",
            "Finding 2: Gross margin expanding",
            "Finding 3: Balance sheet is healthy",
            "Finding 4: FCF conversion high",
            "Finding 5: Piotroski 7/9",
        ],
        "strengths": ["High margins", "Low leverage"],
        "weaknesses": ["Moderate valuation"],
        "business_quality_assessment": "Solid business with durable advantages",
        "financial_health_assessment": "Strong balance sheet",
        "trend_assessment": "Improving margins",
        "moat_indicators": "ROIC well above WACC",
        "red_flags": [],
        "embedded_expectations": "12% FCF CAGR priced in",
        "margin_of_safety_assessment": "Minimal margin of safety",
        "fcf_yield_assessment": "3% yield is modest",
        "historical_valuation_context": "At 70th percentile",
        "reverse_dcf_interpretation": "Growth target is achievable",
        "valuation_risks": ["Multiple compression possible"],
        "valuation_opportunities": ["Earnings beat could re-rate"],
        "base_case_return_expectation": "6-8% annually",
        "growth_quality_assessment": "Fundamental growth confirmed by FCF",
        "revenue_growth_sustainability": "Sustainable driven by pricing power",
        "pricing_power_evidence": "Gross margins stable to improving",
        "scalability_assessment": "High operating leverage visible",
        "management_execution": "Consistent delivery",
        "competitive_advantage_assessment": "Strong switching costs",
        "reinvestment_quality": "ROIC 20% above WACC",
        "growth_risks": ["Market saturation long term"],
        "growth_catalysts": ["New product cycle"],
        "earnings_setup": "Set up to beat on margins",
        "margin_catalyst": "Operating leverage improving",
        "capital_return_catalyst": "Buybacks ongoing",
        "structural_catalysts": ["AI integration opportunity"],
        "near_term_risks": ["Macro headwinds"],
        "guidance_watch": "Watch for FY guidance raise",
        "time_horizon_assessment": "12M: margin expansion",
        "risk_register": {
            "balance_sheet_risk": {"severity": "LOW", "assessment": "0.8x ND/EBITDA", "key_metric": "0.8x"},
            "competitive_risk": {"severity": "MEDIUM", "assessment": "Some competition", "key_metric": "Margins OK"},
            "valuation_compression_risk": {"severity": "MEDIUM", "assessment": "70th pctile", "key_metric": "70th"},
            "earnings_risk": {"severity": "LOW", "assessment": "High quality", "key_metric": "EQ: 0.82"},
            "macro_risk": {"severity": "MEDIUM", "assessment": "Beta 1.1", "key_metric": "Beta 1.1"},
            "concentration_risk": {"severity": "LOW", "assessment": "Diversified", "key_metric": "N/A"},
        },
        "maximum_loss_estimate": "-35% in bear scenario",
        "altman_z_interpretation": "Z=4.5 in safe zone",
        "var_interpretation": "VaR -30% at 95%",
        "top_risks_ranked": ["#1 Valuation", "#2 Macro", "#3 Competition"],
        "risk_mitigants": ["Strong FCF", "Low leverage"],
        "thesis_vulnerabilities": ["12% growth optimistic if macro weakens"],
        "dcf_assumptions_challenged": "Discount rate may be too low",
        "growth_story_challenged": "3Y CAGR only 10%, below current expectations",
        "what_must_go_right": "Revenue acceleration AND margin expansion",
        "historical_analog": "Similar mid-cap SaaS peers",
        "ignored_red_flags": ["FCF growth lagging EPS growth"],
        "probability_weighted_bear_case": "-35% if thesis breaks",
        "what_would_make_bear_wrong": "Revenue reaccelerates to 18%+",
        "recommended_position_size_pct": 4.0,
        "position_sizing_rationale": "4% given moderate conviction",
        "sector_concentration_assessment": "Tech at 25% is manageable",
        "correlation_assessment": "Some overlap with existing tech",
        "risk_contribution_assessment": "Moderate systematic risk",
        "portfolio_fit_verdict": "ACCUMULATE",
        "conditions_for_larger_position": "Price decline to intrinsic value",
        "conditions_for_no_position": "Score drops below 5",
        "alternative_opportunity_cost": "Comparable quality at lower multiples",
    }


def _make_committee_with_mock_agents(scores: dict[str, float]) -> CommitteeEngine:
    """Build a CommitteeEngine where each agent returns a specific score."""
    engine = CommitteeEngine.__new__(CommitteeEngine)
    engine.weights = DEFAULT_WEIGHTS
    engine.portfolio = MagicMock()
    engine.portfolio.existing_positions = {}
    engine.portfolio.sector_weights = {}
    engine.portfolio.target_position_max = 0.10
    engine.portfolio.target_position_min = 0.02
    engine.portfolio.target_sector_max = 0.30
    engine.portfolio.risk_budget_used_pct = 0.0
    engine.portfolio.benchmark = "S&P 500"
    engine.persist_to_db = False
    engine.log_to_journal = False

    def _mock_agent(name: str, score: float):
        agent = MagicMock()
        resp = _make_agent_response(score, summary_prefix=f"{name}: ")
        analysis = MagicMock()
        analysis.score = score
        analysis.confidence = 0.80
        analysis.summary = resp["summary"]
        analysis.key_findings = resp["key_findings"]
        analysis.raw_output = json.dumps(resp)
        agent.analyse.return_value = analysis
        return agent

    engine._fundamental = _mock_agent("FundamentalAnalyst", scores.get("FundamentalAnalyst", 7.0))
    engine._value = _mock_agent("ValueAnalyst", scores.get("ValueAnalyst", 6.0))
    engine._growth = _mock_agent("GrowthAnalyst", scores.get("GrowthAnalyst", 7.0))
    engine._earnings = _mock_agent("EarningsAnalyst", scores.get("EarningsAnalyst", 6.5))
    engine._risk = _mock_agent("RiskManager", scores.get("RiskManager", 6.5))
    engine._bear = _mock_agent("BearAnalyst", scores.get("BearAnalyst", 5.0))
    engine._portfolio = _mock_agent("PortfolioManager", 7.0)

    # Mock thesis synthesis
    engine._synthesis_client = MagicMock()
    engine._synthesis_model_name = "gemini-2.5-flash"
    engine._synthesis_config = MagicMock()
    synth_resp = MagicMock()
    synth_resp.text = json.dumps({
        "thesis": "TEST CORP shows strong fundamentals with durable margins and ROIC above cost of capital. "
                  "Valuation is fair rather than cheap, requiring continued execution to justify. "
                  "The bear case centres on valuation compression if growth disappoints. "
                  "We rate HOLD pending a better entry price.",
        "invalidation_criteria": [
            "INVALIDATION: Gross margin falls below 38% for 2 consecutive quarters -> DOWNGRADE",
            "INVALIDATION: Net Debt/EBITDA exceeds 3x -> REVIEW",
            "INVALIDATION: Revenue growth decelerates below 5% for 2 quarters -> DOWNGRADE",
            "INVALIDATION: Piotroski F-Score drops below 4 -> REVIEW",
            "INVALIDATION: Stock declines >30% without fundamental deterioration -> ADD",
        ]
    })
    engine._synthesis_client.models.generate_content.return_value = synth_resp

    return engine


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCommitteeIntegration:
    """Full committee run integration tests (no live LLM or yfinance)."""

    def test_committee_returns_verdict(self):
        ctx = _make_mock_context()
        engine = _make_committee_with_mock_agents({
            "FundamentalAnalyst": 7.5,
            "ValueAnalyst": 6.0,
            "GrowthAnalyst": 7.0,
            "EarningsAnalyst": 6.5,
            "RiskManager": 6.5,
            "BearAnalyst": 5.5,
        })
        result = engine.run(ctx)

        assert isinstance(result, CommitteeResult)
        assert isinstance(result.verdict, CommitteeVerdict)
        assert result.verdict.ticker == "TEST"
        assert result.verdict.rating in [Rating.BUY, Rating.HOLD, Rating.AVOID]
        assert 0.0 <= result.verdict.weighted_score <= 10.0

    def test_buy_verdict_when_all_agents_high(self):
        ctx = _make_mock_context()
        engine = _make_committee_with_mock_agents({name: 8.5 for name in DEFAULT_WEIGHTS})
        result = engine.run(ctx)
        assert result.verdict.rating == Rating.BUY

    def test_avoid_verdict_when_all_agents_low(self):
        ctx = _make_mock_context()
        engine = _make_committee_with_mock_agents({name: 2.0 for name in DEFAULT_WEIGHTS})
        result = engine.run(ctx)
        assert result.verdict.rating == Rating.AVOID

    def test_hold_verdict_mixed_scores(self):
        ctx = _make_mock_context()
        engine = _make_committee_with_mock_agents({name: 5.0 for name in DEFAULT_WEIGHTS})
        result = engine.run(ctx)
        assert result.verdict.rating == Rating.HOLD

    def test_disagreement_detection_triggered(self):
        ctx = _make_mock_context()
        # High spread: bulls at 9.5, bear at 1.0 → should trigger
        engine = _make_committee_with_mock_agents({
            "FundamentalAnalyst": 9.5,
            "ValueAnalyst":        9.0,
            "GrowthAnalyst":       9.0,
            "EarningsAnalyst":     8.5,
            "RiskManager":         8.5,
            "BearAnalyst":         1.0,   # Bear strongly disagrees
        })
        result = engine.run(ctx)
        assert result.verdict.disagreement_score > 0
        assert "BearAnalyst" in result.verdict.conflicting_agents
        assert result.verdict.disagreement_flag is True

    def test_no_disagreement_when_scores_uniform(self):
        ctx = _make_mock_context()
        engine = _make_committee_with_mock_agents({name: 7.0 for name in DEFAULT_WEIGHTS})
        result = engine.run(ctx)
        assert result.verdict.disagreement_flag is False
        assert len(result.verdict.conflicting_agents) == 0

    def test_verdict_has_invalidation_criteria(self):
        ctx = _make_mock_context()
        engine = _make_committee_with_mock_agents({name: 7.0 for name in DEFAULT_WEIGHTS})
        result = engine.run(ctx)
        assert len(result.verdict.invalidation_criteria) >= 3

    def test_verdict_has_thesis(self):
        ctx = _make_mock_context()
        engine = _make_committee_with_mock_agents({name: 7.0 for name in DEFAULT_WEIGHTS})
        result = engine.run(ctx)
        assert len(result.verdict.thesis) > 50

    def test_bull_bear_summaries_populated(self):
        ctx = _make_mock_context()
        engine = _make_committee_with_mock_agents({name: 7.0 for name in DEFAULT_WEIGHTS})
        result = engine.run(ctx)
        # Both summaries come from agent .summary fields
        assert isinstance(result.verdict.bull_case_summary, str)
        assert isinstance(result.verdict.bear_case_summary, str)

    def test_agent_confidences_in_verdict(self):
        ctx = _make_mock_context()
        engine = _make_committee_with_mock_agents({name: 7.0 for name in DEFAULT_WEIGHTS})
        result = engine.run(ctx)
        assert len(result.verdict.agent_confidences) == len(DEFAULT_WEIGHTS)

    def test_runtime_tracked(self):
        ctx = _make_mock_context()
        engine = _make_committee_with_mock_agents({name: 7.0 for name in DEFAULT_WEIGHTS})
        result = engine.run(ctx)
        assert result.total_runtime_seconds > 0


class TestDisagreementAlgorithm:
    """Unit tests for the disagreement detection logic."""

    def _engine(self):
        engine = CommitteeEngine.__new__(CommitteeEngine)
        engine.weights = DEFAULT_WEIGHTS
        return engine

    def test_uniform_scores_zero_disagreement(self):
        engine = self._engine()
        scores = {name: 7.0 for name in DEFAULT_WEIGHTS}
        d_score, conflicting = engine._compute_disagreement(scores, 7.0)
        assert d_score == pytest.approx(0.0, abs=0.01)
        assert conflicting == []

    def test_high_spread_high_disagreement(self):
        engine = self._engine()
        scores = {
            "FundamentalAnalyst": 9.5,
            "ValueAnalyst":        9.0,
            "GrowthAnalyst":       9.0,
            "EarningsAnalyst":     9.0,
            "RiskManager":         9.0,
            "BearAnalyst":         1.0,
        }
        weighted = 7.8  # approximate
        d_score, conflicting = engine._compute_disagreement(scores, weighted)
        assert d_score > 3.0
        assert "BearAnalyst" in conflicting

    def test_conflicting_threshold(self):
        engine = self._engine()
        scores = {"A": 7.0, "B": 7.0, "C": 2.0}
        weighted = 6.5
        _, conflicting = engine._compute_disagreement(scores, weighted)
        assert "C" in conflicting
        assert "A" not in conflicting

    def test_disagreement_capped_at_10(self):
        engine = self._engine()
        scores = {"A": 10.0, "B": 0.0}
        d_score, _ = engine._compute_disagreement(scores, 5.0)
        assert d_score <= 10.0

    def test_single_agent_no_disagreement(self):
        engine = self._engine()
        d_score, conflicting = engine._compute_disagreement({"A": 7.5}, 7.5)
        assert d_score == 0.0
        assert conflicting == []


class TestReportFormatter:
    """Tests for the report formatter output."""

    def _make_verdict(self, rating=Rating.HOLD, score=6.5, disagreement=False) -> CommitteeVerdict:
        return CommitteeVerdict(
            ticker="TEST",
            analysis_date=date.today(),
            horizon=Horizon.TWELVE_MONTHS,
            rating=rating,
            weighted_score=score,
            confidence=0.75,
            scenario_model=_make_scenario_model(),
            thesis="TEST CORP is a high-quality business trading at fair value. "
                   "The bull case requires continued execution, while the bear case "
                   "is driven by valuation compression. We rate HOLD.",
            invalidation_criteria=[
                "INVALIDATION: Gross margin < 38% -> DOWNGRADE",
                "INVALIDATION: Net Debt/EBITDA > 3x -> REVIEW",
                "INVALIDATION: Revenue growth < 5% -> REVIEW",
                "INVALIDATION: Piotroski < 4 -> REVIEW",
                "INVALIDATION: Price down 30% -> ADD",
            ],
            agent_scores={
                "FundamentalAnalyst": 7.5, "ValueAnalyst": 6.0,
                "GrowthAnalyst": 7.0, "EarningsAnalyst": 6.5,
                "RiskManager": 6.5, "BearAnalyst": 5.5,
            },
            agent_confidences={
                "FundamentalAnalyst": 0.85, "ValueAnalyst": 0.80,
                "GrowthAnalyst": 0.75, "EarningsAnalyst": 0.70,
                "RiskManager": 0.80, "BearAnalyst": 0.75,
            },
            disagreement_score=2.1 if not disagreement else 4.5,
            conflicting_agents=[] if not disagreement else ["BearAnalyst"],
            disagreement_flag=disagreement,
            bull_case_summary="Strong fundamentals and pricing power support the business.",
            bear_case_summary="Valuation leaves no margin of safety if growth disappoints.",
        )

    def _make_mock_result(self, verdict: CommitteeVerdict) -> MagicMock:
        result = MagicMock()
        result.verdict = verdict
        result.agent_analyses = {
            "FundamentalAnalyst": MagicMock(
                score=7.5, confidence=0.85, summary="Strong business quality.",
                key_findings=["ROIC 20% > WACC", "Margins expanding"]
            ),
            "BearAnalyst": MagicMock(
                score=5.5, confidence=0.75, summary="Valuation leaves no cushion.",
                key_findings=["18% growth priced in", "FCF declining"]
            ),
        }
        result.portfolio_analysis = MagicMock(
            score=7.0, confidence=0.75,
            summary="Good portfolio fit at 4% position.",
            key_findings=["4% position", "ACCUMULATE", "Sector weight manageable"]
        )
        result.agent_scores = verdict.agent_scores
        result.agent_confidences = verdict.agent_confidences
        result.total_runtime_seconds = 42.5
        return result

    def test_terminal_report_contains_ticker(self):
        from finance_agent.reporting.report import ReportFormatter
        verdict = self._make_verdict()
        result = self._make_mock_result(verdict)
        fmt = ReportFormatter(result)
        lines = fmt._build_terminal_lines(use_colour=False)
        full = "\n".join(lines)
        assert "TEST" in full

    def test_terminal_report_contains_rating(self):
        from finance_agent.reporting.report import ReportFormatter
        verdict = self._make_verdict(rating=Rating.BUY, score=7.5)
        result = self._make_mock_result(verdict)
        fmt = ReportFormatter(result)
        lines = fmt._build_terminal_lines(use_colour=False)
        full = "\n".join(lines)
        assert "BUY" in full

    def test_disagreement_flag_shown_in_report(self):
        from finance_agent.reporting.report import ReportFormatter
        verdict = self._make_verdict(disagreement=True)
        result = self._make_mock_result(verdict)
        fmt = ReportFormatter(result)
        lines = fmt._build_terminal_lines(use_colour=False)
        full = "\n".join(lines)
        assert "SPLIT COMMITTEE" in full or "DISAGREEMENT" in full

    def test_no_disagreement_flag_when_unanimous(self):
        from finance_agent.reporting.report import ReportFormatter
        verdict = self._make_verdict(disagreement=False)
        result = self._make_mock_result(verdict)
        fmt = ReportFormatter(result)
        lines = fmt._build_terminal_lines(use_colour=False)
        full = "\n".join(lines)
        assert "SPLIT COMMITTEE" not in full

    def test_json_export_valid(self):
        from finance_agent.reporting.report import ReportFormatter
        verdict = self._make_verdict()
        result = self._make_mock_result(verdict)
        fmt = ReportFormatter(result)
        data = fmt.to_dict()
        assert data["ticker"] == "TEST"
        assert data["rating"] == "HOLD"
        assert "scenarios" in data
        assert "invalidation_criteria" in data
        assert len(data["invalidation_criteria"]) == 5

    def test_json_file_export(self):
        from finance_agent.reporting.report import ReportFormatter
        verdict = self._make_verdict()
        result = self._make_mock_result(verdict)
        fmt = ReportFormatter(result)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = fmt.export_json(Path(tmpdir) / "test_report.json")
            assert path.exists()
            content = json.loads(path.read_text())
            assert content["ticker"] == "TEST"

    def test_markdown_file_export(self):
        from finance_agent.reporting.report import ReportFormatter
        verdict = self._make_verdict()
        result = self._make_mock_result(verdict)
        fmt = ReportFormatter(result)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = fmt.export_markdown(Path(tmpdir) / "test_report.md")
            assert path.exists()
            content = path.read_text()
            assert "TEST" in content

    def test_rating_label_includes_split_flag(self):
        verdict = self._make_verdict(disagreement=True)
        assert "SPLIT COMMITTEE" in verdict.rating_label

    def test_ev_formatted(self):
        verdict = self._make_verdict()
        assert "%" in verdict.ev_formatted


class TestCLI:
    """Basic CLI argument parsing tests (no live execution)."""

    def test_parser_analyze_command(self):
        from finance_agent.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["analyze", "AAPL", "--horizon", "12M"])
        assert args.command == "analyze"
        assert "AAPL" in args.tickers
        assert args.horizon == "12M"

    def test_parser_multiple_tickers(self):
        from finance_agent.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["analyze", "AAPL", "MSFT", "NVDA"])
        assert len(args.tickers) == 3

    def test_parser_no_persist_flag(self):
        from finance_agent.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["analyze", "AAPL", "--no-persist"])
        assert args.persist is False

    def test_parser_journal_stats(self):
        from finance_agent.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["journal", "stats"])
        assert args.command == "journal"
        assert args.subcommand == "stats"

    def test_parser_journal_pending(self):
        from finance_agent.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["journal", "pending"])
        assert args.subcommand == "pending"

    def test_parser_json_flag(self):
        from finance_agent.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["analyze", "AAPL", "--json"])
        assert args.json is True
