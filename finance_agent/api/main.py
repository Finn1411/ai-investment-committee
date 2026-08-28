from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yfinance as yf
from datetime import datetime
import json
import os

from finance_agent.database.db import get_session, StockORM, PredictionORM
from finance_agent.agents.committee import CommitteeEngine
from finance_agent.agents.portfolio_manager import PortfolioContext
from finance_agent.data.pipeline import DataPipeline
from finance_agent.models.schemas import Horizon
from finance_agent.evaluation.journal import PredictionJournal
from finance_agent.evaluation.backtest import BacktestEngine
from finance_agent.reporting.report import ReportFormatter
from finance_agent.utils.logger import setup_logger

setup_logger()

app = FastAPI(title="Finance Agent API", version="1.0.0")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    horizon: str = "12M"
    persist: bool = True

@app.get("/api/watchlist")
def get_watchlist():
    """Get all tracked stocks"""
    with get_session() as session:
        stocks = session.query(StockORM).all()
        return [{"ticker": s.ticker, "name": s.name, "sector": s.sector, "industry": s.industry, "added_at": s.added_at} for s in stocks]

@app.post("/api/watchlist/{ticker}")
def add_to_watchlist(ticker: str):
    """Add a stock to watchlist"""
    ticker = ticker.upper()
    with get_session() as session:
        existing = session.query(StockORM).filter_by(ticker=ticker).first()
        if existing:
            return {"status": "already_exists", "ticker": ticker}
        
        # Fetch metadata using yfinance
        try:
            info = yf.Ticker(ticker).info
            name = info.get("shortName", ticker)
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")
            country = info.get("country", "Unknown")
            currency = info.get("currency", "USD")
            exchange = info.get("exchange", "Unknown")
        except Exception:
            name, sector, industry, country, currency, exchange = ticker, "Unknown", "Unknown", "Unknown", "USD", "Unknown"

        stock = StockORM(
            ticker=ticker,
            name=name,
            sector=sector,
            industry=industry,
            country=country,
            currency=currency,
            exchange=exchange,
            added_at=datetime.utcnow()
        )
        session.add(stock)
        return {"status": "added", "ticker": ticker, "name": name}

@app.delete("/api/watchlist/{ticker}")
def remove_from_watchlist(ticker: str):
    """Remove a stock from watchlist"""
    ticker = ticker.upper()
    with get_session() as session:
        stock = session.query(StockORM).filter_by(ticker=ticker).first()
        if not stock:
            raise HTTPException(status_code=404, detail="Stock not found in watchlist")
        session.delete(stock)
        return {"status": "removed", "ticker": ticker}

@app.post("/api/analyze/{ticker}")
def analyze_stock(ticker: str, req: AnalyzeRequest):
    """Run the complete pipeline and committee analysis on a ticker"""
    ticker = ticker.upper()
    
    horizon_map = {
        "3M":   Horizon.THREE_MONTHS,
        "12M":  Horizon.TWELVE_MONTHS,
        "3-5Y": Horizon.THREE_FIVE_YEARS,
    }
    horizon = horizon_map.get(req.horizon, Horizon.TWELVE_MONTHS)
    
    portfolio = PortfolioContext()
    pipeline = DataPipeline(horizon=horizon, persist_to_db=req.persist)
    committee = CommitteeEngine(portfolio=portfolio, persist_to_db=req.persist, log_to_journal=req.persist)
    
    try:
        context = pipeline.run(ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data pipeline failed: {str(e)}")
        
    try:
        result = committee.run(context, horizon=horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Committee analysis failed: {str(e)}")
        
    formatter = ReportFormatter(result)
    return formatter.to_dict()

@app.get("/api/watchlist/last-analyses")
def get_last_analyses():
    """Get the most recent prediction for each watchlisted ticker."""
    with get_session() as session:
        stocks = session.query(StockORM).all()
        result = {}
        for s in stocks:
            latest = (
                session.query(PredictionORM)
                .filter_by(ticker=s.ticker)
                .order_by(PredictionORM.created_at.desc())
                .first()
            )
            if latest:
                committee = json.loads(latest.committee_verdict_json) if latest.committee_verdict_json else {}
                result[s.ticker] = {
                    "rating": latest.rating,
                    "score": committee.get("weighted_score"),
                    "date": latest.analysis_date,
                    "horizon": latest.horizon,
                }
        return result

@app.get("/api/journal/stats")
def get_journal_stats():
    """Get system backtesting and calibration statistics"""
    engine = BacktestEngine()
    stats = engine.get_stats()
    
    journal = PredictionJournal()
    j_stats = journal.summary_stats()
    
    # Merge stats
    stats["total_predictions"] = j_stats["total_predictions"]
    stats["pending_reviews"] = len(journal.get_pending_reviews())
    
    return stats

@app.get("/api/journal/predictions")
def get_predictions():
    """Get history of all predictions"""
    with get_session() as session:
        preds = session.query(PredictionORM).order_by(PredictionORM.created_at.desc()).limit(100).all()
        result = []
        for p in preds:
            result.append({
                "id": p.prediction_id,
                "seq": p.sequence_number,
                "ticker": p.ticker,
                "date": p.analysis_date,
                "rating": p.rating,
                "confidence": p.confidence,
                "actual_return": p.actual_return,
                "review_status": p.review_status,
                "expected_return": p.expected_return
            })
        return result

@app.get("/api/journal/predictions/{id}")
def get_prediction_detail(id: str):
    """Get full details of a past prediction to render in the UI"""
    with get_session() as session:
        p = session.query(PredictionORM).filter_by(prediction_id=id).first()
        if not p:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        committee = json.loads(p.committee_verdict_json) if p.committee_verdict_json else {}
        scenarios = json.loads(p.scenario_model_json) if p.scenario_model_json else {}
        risks = json.loads(p.invalidation_criteria) if p.invalidation_criteria else []
        
        # Translate to the format expected by renderReport
        return {
            "ticker": p.ticker,
            "horizon": p.horizon,
            "weighted_score": committee.get("weighted_score", 0),
            "rating": p.rating,
            "thesis": p.thesis or committee.get("synthesized_thesis", ""),
            "scenarios": scenarios,
            "invalidation_criteria": risks,
            "disagreement": {
                "score": committee.get("disagreement_score", 0)
            },
            "agent_outputs": {} # History doesn't store full agent outputs yet
        }

# Serve static files for the frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
