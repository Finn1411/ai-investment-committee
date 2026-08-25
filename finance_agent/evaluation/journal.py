"""
Prediction Journal — immutable log for every investment decision.
Implements Phase 7 of the masterplan.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from finance_agent.database.db import (
    PredictionORM,
    PredictionReviewORM,
    get_session,
    init_db,
)
from finance_agent.models.predictions import PredictionEntry, PredictionReview, ReviewStatus
from finance_agent.utils.logger import logger


class PredictionJournal:
    """
    Writes and reads prediction entries from the SQLite database.
    Entries are immutable once written.
    """

    def __init__(self) -> None:
        init_db()

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(self, entry: PredictionEntry) -> PredictionEntry:
        """Persist a new prediction. Assigns sequence number automatically."""
        with get_session() as session:
            # Auto-increment sequence number
            last = (
                session.query(PredictionORM.sequence_number)
                .order_by(PredictionORM.sequence_number.desc())
                .first()
            )
            seq = (last[0] or 0) + 1 if last else 1
            entry.sequence_number = seq

            orm = PredictionORM(
                prediction_id=entry.prediction_id,
                sequence_number=seq,
                ticker=entry.ticker,
                analysis_date=str(entry.analysis_date),
                horizon=entry.horizon.value,
                rating=entry.rating.value,
                expected_return=entry.expected_return,
                confidence=entry.confidence,
                thesis=entry.thesis,
                key_risks=json.dumps(entry.key_risks),
                invalidation_criteria=json.dumps(entry.invalidation_criteria),
                scenario_model_json=entry.scenario_model.model_dump_json(),
                committee_verdict_json=(
                    entry.committee_verdict.model_dump_json()
                    if entry.committee_verdict
                    else None
                ),
                review_status=entry.review_status.value,
            )
            session.add(orm)
            session.commit()

        logger.info(f"[Journal] Recorded: {entry.label}")
        return entry

    # ── Review ────────────────────────────────────────────────────────────────

    def add_review(self, prediction_id: str, review: PredictionReview) -> None:
        """Append a review to an existing prediction (non-destructive)."""
        with get_session() as session:
            pred = session.query(PredictionORM).filter_by(prediction_id=prediction_id).first()
            if not pred:
                raise ValueError(f"Prediction {prediction_id} not found in journal.")

            rev_orm = PredictionReviewORM(
                prediction_id=prediction_id,
                review_date=str(review.review_date),
                actual_return=review.actual_return,
                benchmark_return=review.benchmark_return,
                alpha=review.alpha,
                rating_correct=review.rating_correct,
                notes=review.notes,
            )
            session.add(rev_orm)

            # Update status
            existing = session.query(PredictionReviewORM).filter_by(
                prediction_id=prediction_id
            ).count()
            if existing == 0:
                pred.review_status = ReviewStatus.REVIEWED_3M.value
            elif existing == 1:
                pred.review_status = ReviewStatus.REVIEWED_6M.value
            else:
                pred.review_status = ReviewStatus.REVIEWED_12M.value

            session.commit()
        logger.info(f"[Journal] Added review for prediction {prediction_id[:8]}...")

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_pending_reviews(self, as_of: Optional[date] = None) -> list[dict]:
        """Return predictions whose review date has passed but have no review yet."""
        as_of = as_of or date.today()
        results = []
        with get_session() as session:
            rows = (
                session.query(PredictionORM)
                .filter(PredictionORM.review_status == ReviewStatus.PENDING.value)
                .all()
            )
            for row in rows:
                analysis_date = date.fromisoformat(row.analysis_date)
                # Rough check: if 3M horizon, review after ~90 days
                days_elapsed = (as_of - analysis_date).days
                if days_elapsed >= 85:
                    results.append({
                        "prediction_id": row.prediction_id,
                        "seq": row.sequence_number,
                        "ticker": row.ticker,
                        "analysis_date": row.analysis_date,
                        "horizon": row.horizon,
                        "days_elapsed": days_elapsed,
                    })
        return results

    def summary_stats(self) -> dict:
        """Basic statistics across all predictions."""
        with get_session() as session:
            total = session.query(PredictionORM).count()
            reviewed = (
                session.query(PredictionReviewORM).count()
            )
            correct = (
                session.query(PredictionReviewORM)
                .filter(PredictionReviewORM.rating_correct == True)  # noqa: E712
                .count()
            )
            hit_rate = correct / reviewed if reviewed > 0 else None
        return {
            "total_predictions": total,
            "total_reviews": reviewed,
            "correct_calls": correct,
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        }

    # ── Convenience ───────────────────────────────────────────────────────────

    def log(
        self,
        ticker: str,
        analysis_date: date,
        horizon,
        rating,
        weighted_score: float,
        confidence: float,
        scenario_model,
        thesis: str,
        invalidation_criteria: list[str],
        agent_scores: dict[str, float],
        metrics_snapshot: dict | None = None,
    ) -> None:
        """
        Convenience wrapper used by CommitteeEngine.
        Builds a PredictionEntry and records it.
        """
        from finance_agent.models.predictions import PredictionEntry, ReviewStatus
        entry = PredictionEntry(
            ticker=ticker,
            analysis_date=analysis_date,
            horizon=horizon,
            rating=rating,
            expected_return=float(scenario_model.expected_value),
            confidence=confidence,
            thesis=thesis,
            key_risks=invalidation_criteria,
            invalidation_criteria=invalidation_criteria,
            scenario_model=scenario_model,
            review_status=ReviewStatus.PENDING,
        )
        self.record(entry)

