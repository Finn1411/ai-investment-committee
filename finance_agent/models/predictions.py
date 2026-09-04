"""
Prediction Journal schema + helpers.
Every investment decision gets an immutable ID and is tracked through its life.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from finance_agent.models.schemas import (
    CommitteeVerdict,
    Horizon,
    Rating,
    ScenarioModel,
)


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    REVIEWED_3M = "REVIEWED_3M"
    REVIEWED_6M = "REVIEWED_6M"
    REVIEWED_12M = "REVIEWED_12M"
    CLOSED = "CLOSED"


class PredictionReview(BaseModel):
    """Filled in after the review date passes."""
    review_date: date
    actual_return: float                    # Realised return of the stock
    benchmark_return: float                 # Benchmark return over same period
    alpha: float                            # actual_return - benchmark_return
    rating_correct: bool                    # Did direction match?
    notes: Optional[str] = None


class PredictionEntry(BaseModel):
    """
    Immutable prediction record — written once, never changed.
    Reviews are appended separately.
    """
    prediction_id: str = Field(default_factory=lambda: str(uuid4()))
    sequence_number: Optional[int] = None  # Human-readable counter
    ticker: str
    analysis_date: date
    horizon: Horizon
    rating: Rating
    expected_return: float
    weighted_score: Optional[float] = None  # Committee composite score (0-10)
    confidence: float = Field(ge=0.0, le=1.0)
    scenario_model: ScenarioModel
    thesis: str
    key_risks: list[str] = Field(default_factory=list)
    invalidation_criteria: list[str] = Field(default_factory=list)
    committee_verdict: Optional[CommitteeVerdict] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Review data (populated later)
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviews: list[PredictionReview] = Field(default_factory=list)
    actual_return: Optional[float] = None
    resolved_at: Optional[datetime] = None

    @property
    def label(self) -> str:
        seq = f"#{self.sequence_number:05d}" if self.sequence_number else self.prediction_id[:8]
        return f"PREDICTION {seq} | {self.ticker} | {self.analysis_date}"
