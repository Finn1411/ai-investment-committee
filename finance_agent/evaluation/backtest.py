from __future__ import annotations

import yfinance as yf
from datetime import date, datetime
from typing import Optional

from finance_agent.database.db import get_session, PredictionORM, PredictionReviewORM
from finance_agent.models.predictions import PredictionReview, ReviewStatus
from finance_agent.utils.logger import logger
from finance_agent.utils.config import settings

class BacktestEngine:
    """
    Engine to evaluate predictions and calculate system calibration metrics (Brier score, etc).
    """

    def resolve_prediction(self, prediction_id: str, actual_return: Optional[float] = None) -> PredictionReview:
        """
        Resolves a prediction by calculating its actual return and appending a review.
        If actual_return is None, fetches historical prices.
        """
        with get_session() as session:
            pred = session.query(PredictionORM).filter_by(prediction_id=prediction_id).first()
            if not pred:
                # Also try matching by ticker for convenience if it's not a UUID
                pred = session.query(PredictionORM).filter_by(ticker=prediction_id.upper(), review_status=ReviewStatus.PENDING.value).first()
                if not pred:
                    raise ValueError(f"Pending prediction not found for {prediction_id}")
            
            ticker = pred.ticker
            analysis_date_str = pred.analysis_date
            
            start_date = analysis_date_str
            end_date = str(date.today())
            
            if actual_return is None:
                # Fetch ticker data
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if stock_data.empty:
                    raise ValueError(f"Could not fetch price data for {ticker}")
                
                start_price = float(stock_data['Close'].values.flatten()[0])
                end_price = float(stock_data['Close'].values.flatten()[-1])
                actual_return = (end_price - start_price) / start_price

            # Benchmark return
            benchmark = settings.benchmark
            bench_data = yf.download(benchmark, start=start_date, end=end_date, progress=False)
            if bench_data.empty:
                logger.warning(f"Could not fetch benchmark {benchmark}, assuming 0% return")
                benchmark_return = 0.0
            else:
                b_start = float(bench_data['Close'].values.flatten()[0])
                b_end = float(bench_data['Close'].values.flatten()[-1])
                benchmark_return = (b_end - b_start) / b_start

            alpha = actual_return - benchmark_return
            
            # Determine if rating was correct
            rating = pred.rating
            rating_correct = False
            if rating == "BUY" and alpha > 0:
                rating_correct = True
            elif rating == "AVOID" and alpha < 0:
                rating_correct = True
            elif rating == "HOLD" and abs(alpha) < 0.05: # within 5% of benchmark
                rating_correct = True

            review = PredictionReview(
                review_date=date.today(),
                actual_return=actual_return,
                benchmark_return=benchmark_return,
                alpha=alpha,
                rating_correct=rating_correct,
                notes="Resolved via BacktestEngine"
            )

            # Update prediction
            pred.actual_return = actual_return
            pred.resolved_at = datetime.utcnow()
            pred.review_status = ReviewStatus.CLOSED.value

            # Add review entry
            rev_orm = PredictionReviewORM(
                prediction_id=pred.prediction_id,
                review_date=str(review.review_date),
                actual_return=review.actual_return,
                benchmark_return=review.benchmark_return,
                alpha=review.alpha,
                rating_correct=review.rating_correct,
                notes=review.notes,
            )
            session.add(rev_orm)
            session.commit()
            
            logger.info(f"[BacktestEngine] Resolved {ticker} (ID: {pred.prediction_id[:8]}) -> Actual: {actual_return:+.2%}, Alpha: {alpha:+.2%}")
            return review

    def get_stats(self) -> dict:
        """
        Calculate Brier Score and Hit Rate.
        """
        with get_session() as session:
            preds = session.query(PredictionORM).filter(PredictionORM.review_status == ReviewStatus.CLOSED.value).all()
            
            if not preds:
                return {"total_resolved": 0}

            total = len(preds)
            correct_calls = 0
            brier_sum = 0.0
            
            # Hit rate by rating
            stats_by_rating = {"BUY": {"total": 0, "correct": 0}, "HOLD": {"total": 0, "correct": 0}, "AVOID": {"total": 0, "correct": 0}}
            
            for p in preds:
                # Is it correct? We consider rating correctness.
                # Let's see if the alpha was positive or negative to evaluate confidence vs outcome
                # Brier score calculation: (confidence - outcome)^2
                # If outcome > benchmark, outcome = 1, else 0
                outcome = 1.0 if (p.actual_return and p.actual_return > p.benchmark_return if hasattr(p, 'benchmark_return') else p.actual_return > 0) else 0.0
                
                # Fetch review for benchmark
                review = session.query(PredictionReviewORM).filter_by(prediction_id=p.prediction_id).order_by(PredictionReviewORM.id.desc()).first()
                if review:
                    outcome = 1.0 if p.actual_return and p.actual_return > review.benchmark_return else 0.0
                
                # If BUY, confidence is P(outperform). If AVOID, confidence is P(underperform)
                prob = p.confidence
                if p.rating == "AVOID":
                    prob = 1.0 - p.confidence
                elif p.rating == "HOLD":
                    prob = 0.5 # Neutral
                    
                brier_sum += (prob - outcome) ** 2
                
                if review and review.rating_correct:
                    correct_calls += 1
                    stats_by_rating[p.rating]["correct"] += 1
                    
                stats_by_rating[p.rating]["total"] += 1
                
            brier_score = brier_sum / total
            hit_rate = correct_calls / total
            
            # Formatting stats
            for r in stats_by_rating:
                r_total = stats_by_rating[r]["total"]
                stats_by_rating[r]["hit_rate"] = stats_by_rating[r]["correct"] / r_total if r_total > 0 else 0.0

            return {
                "total_resolved": total,
                "hit_rate": hit_rate,
                "brier_score": brier_score,
                "by_rating": stats_by_rating
            }
