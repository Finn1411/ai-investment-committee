"""Pydantic schemas for all core data structures in the Finance Agent."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────

class Rating(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID"
    UNDER_REVIEW = "UNDER_REVIEW"


class Horizon(str, Enum):
    THREE_MONTHS = "3M"
    TWELVE_MONTHS = "12M"
    THREE_FIVE_YEARS = "3-5Y"


class DataTier(str, Enum):
    """Source quality tier for RAG / claims (Phase 8)."""
    TIER1_PRIMARY = "TIER1"       # Annual/quarterly reports, filings
    TIER2_SECONDARY = "TIER2"     # Reputable financial news, industry reports
    TIER3_OPINION = "TIER3"       # Analyst opinions, blogs, social media


class DataStatus(str, Enum):
    REPORTED = "REPORTED"
    ESTIMATED = "ESTIMATED"
    DERIVED = "DERIVED"


# ── Market Data ───────────────────────────────────────────────────────────────

class OHLCVRecord(BaseModel):
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None


class MarketMetrics(BaseModel):
    ticker: str
    as_of: date
    return_1d: Optional[float] = None
    return_1w: Optional[float] = None
    return_1m: Optional[float] = None
    return_3m: Optional[float] = None
    return_6m: Optional[float] = None
    return_1y: Optional[float] = None
    return_ytd: Optional[float] = None
    volatility_30d: Optional[float] = None
    volatility_90d: Optional[float] = None
    beta: Optional[float] = None
    max_drawdown_1y: Optional[float] = None
    relative_strength_vs_benchmark: Optional[float] = None
    avg_daily_volume_30d: Optional[float] = None
    market_cap: Optional[float] = None


# ── Fundamentals ──────────────────────────────────────────────────────────────

class FundamentalMetrics(BaseModel):
    ticker: str
    fiscal_year: int
    fiscal_quarter: Optional[int] = None
    report_date: Optional[date] = None
    data_status: DataStatus = DataStatus.REPORTED

    # Profitability
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roic: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None

    # Cash Flow
    free_cash_flow: Optional[float] = None
    fcf_margin: Optional[float] = None
    fcf_yield: Optional[float] = None
    cash_conversion_ratio: Optional[float] = None

    # Balance Sheet
    net_debt_to_ebitda: Optional[float] = None
    interest_coverage: Optional[float] = None
    current_ratio: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None

    # Growth (YoY %)
    revenue_growth_yoy: Optional[float] = None
    eps_growth_yoy: Optional[float] = None
    fcf_growth_yoy: Optional[float] = None

    # Revenue / Earnings absolute
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    ebit: Optional[float] = None
    net_income: Optional[float] = None
    eps_diluted: Optional[float] = None


class ValuationMetrics(BaseModel):
    ticker: str
    as_of: date
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    ev_to_ebit: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    price_to_sales: Optional[float] = None
    price_to_book: Optional[float] = None
    fcf_yield: Optional[float] = None
    enterprise_value: Optional[float] = None
    # Historical percentile (0-100) versus own history
    pe_percentile_5y: Optional[float] = None
    ev_ebitda_percentile_5y: Optional[float] = None


# ── Scenarios ─────────────────────────────────────────────────────────────────

class Scenario(BaseModel):
    """One scenario leg (Bear / Base / Bull)."""
    label: str                          # "Bear", "Base", "Bull"
    probability: float = Field(ge=0.0, le=1.0)
    expected_return: float              # e.g. -0.28 for -28%
    narrative: Optional[str] = None     # Short textual description

    @field_validator("probability")
    @classmethod
    def round_prob(cls, v: float) -> float:
        return round(v, 4)


class ScenarioModel(BaseModel):
    ticker: str
    horizon: Horizon
    analysis_date: date
    bear: Scenario
    base: Scenario
    bull: Scenario

    @property
    def expected_value(self) -> float:
        return (
            self.bear.probability * self.bear.expected_return
            + self.base.probability * self.base.expected_return
            + self.bull.probability * self.bull.expected_return
        )

    @property
    def prob_outperform(self) -> Optional[float]:
        """Probability of positive absolute return (simplified proxy)."""
        positive = sum(
            s.probability for s in [self.bear, self.base, self.bull]
            if s.expected_return > 0
        )
        return round(positive, 4)


# ── Agent Output ──────────────────────────────────────────────────────────────

class AgentAnalysis(BaseModel):
    agent_name: str
    ticker: str
    analysis_date: date
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    score: Optional[float] = Field(None, ge=0.0, le=10.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    raw_output: Optional[str] = None
    sources: list[str] = Field(default_factory=list)


class CommitteeVerdict(BaseModel):
    ticker: str
    analysis_date: date
    horizon: Horizon
    rating: Rating
    weighted_score: float = Field(ge=0.0, le=10.0)
    confidence: float = Field(ge=0.0, le=1.0)
    scenario_model: ScenarioModel
    thesis: str
    invalidation_criteria: list[str] = Field(default_factory=list)
    agent_scores: dict[str, float] = Field(default_factory=dict)
    agent_confidences: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # -- Disagreement Detection (Week 4) ----------------------------------------
    disagreement_score: float = Field(
        default=0.0, ge=0.0, le=10.0,
        description="Std dev of agent scores * 2. >3.0 = high disagreement."
    )
    conflicting_agents: list[str] = Field(
        default_factory=list,
        description="Agents whose score diverges >2.5 pts from the weighted score."
    )
    disagreement_flag: bool = Field(
        default=False,
        description="True if disagreement_score > 3.0 — committee is split."
    )

    # -- Narrative summaries (Week 4) -------------------------------------------
    bull_case_summary: str = Field(
        default="", description="1-2 sentence bull case from positive agents."
    )
    bear_case_summary: str = Field(
        default="", description="1-2 sentence bear case from BearAnalyst."
    )

    @property
    def rating_label(self) -> str:
        flag = " [SPLIT COMMITTEE]" if self.disagreement_flag else ""
        return f"{self.rating.value}{flag}"

    @property
    def ev_formatted(self) -> str:
        ev = self.scenario_model.expected_value
        return f"{ev:+.1%}"
