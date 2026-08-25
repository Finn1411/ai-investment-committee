import pytest
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock
from finance_agent.database.db import get_session, PredictionORM, PredictionReviewORM, init_db
from finance_agent.evaluation.backtest import BacktestEngine
from finance_agent.models.predictions import ReviewStatus
from finance_agent.utils.config import settings

@pytest.fixture
def test_db():
    init_db()
    
@pytest.fixture
def mock_prediction(test_db):
    with get_session() as session:
        # clear existing
        session.query(PredictionReviewORM).delete()
        session.query(PredictionORM).delete()
        
        pred = PredictionORM(
            prediction_id="test-1",
            sequence_number=1,
            ticker="TEST_TICKER",
            analysis_date=str(date.today() - timedelta(days=90)),
            horizon="12M",
            rating="BUY",
            expected_return=0.15,
            confidence=0.8,
            thesis="Test thesis",
            key_risks="[]",
            invalidation_criteria="[]",
            scenario_model_json="{}",
            review_status=ReviewStatus.PENDING.value
        )
        session.add(pred)
        session.commit()
    return "test-1"

@patch("finance_agent.evaluation.backtest.yf.download")
def test_resolve_prediction_explicit(mock_yf_download, mock_prediction):
    engine = BacktestEngine()
    
    # We resolve explicitly, yfinance shouldn't be called for the stock, 
    # but it WILL be called for the benchmark unless we mock it or pass something.
    # We mock yfinance entirely to return a dummy dataframe just in case.
    import pandas as pd
    import numpy as np
    dummy_df = pd.DataFrame({"Close": [100.0, 110.0]})
    mock_yf_download.return_value = dummy_df
    
    review = engine.resolve_prediction(mock_prediction, actual_return=0.12)
    
    assert review.actual_return == 0.12
    # Benchmark return should be (110 - 100)/100 = 0.1
    assert abs(review.benchmark_return - 0.1) < 1e-6
    assert abs(review.alpha - 0.02) < 1e-6
    assert review.rating_correct is True

def test_get_stats(mock_prediction):
    engine = BacktestEngine()
    
    # Manually resolve the prediction
    with get_session() as session:
        pred = session.query(PredictionORM).filter_by(prediction_id=mock_prediction).first()
        pred.actual_return = 0.15
        pred.resolved_at = datetime.utcnow()
        pred.review_status = ReviewStatus.CLOSED.value
        
        rev = PredictionReviewORM(
            prediction_id=pred.prediction_id,
            review_date=str(date.today()),
            actual_return=0.15,
            benchmark_return=0.05,
            alpha=0.10,
            rating_correct=True,
            notes="Test"
        )
        session.add(rev)
        session.commit()
        
    stats = engine.get_stats()
    
    assert stats["total_resolved"] == 1
    assert stats["hit_rate"] == 1.0
    
    # Brier score = (confidence - outcome)^2
    # outcome for positive alpha = 1.0
    # confidence = 0.8
    # (0.8 - 1.0)^2 = 0.04
    assert abs(stats["brier_score"] - 0.04) < 1e-6
    assert stats["by_rating"]["BUY"]["correct"] == 1
    assert stats["by_rating"]["BUY"]["hit_rate"] == 1.0
