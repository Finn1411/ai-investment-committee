from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
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
from finance_agent.screener.engine import ScreenerEngine
from finance_agent.screener.indices import INDEX_REGISTRY, get_index_tickers
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
        return [{"ticker": s.ticker, "name": s.name, "sector": s.sector, "industry": s.industry, "added_at": str(s.added_at)} for s in stocks]

@app.get("/api/watchlist/prices")
def get_watchlist_prices():
    """Batch-fetch live prices for all watchlisted tickers (single yfinance call)."""
    with get_session() as session:
        stocks = session.query(StockORM).all()
        tickers = [s.ticker for s in stocks]

    if not tickers:
        return {}

    try:
        import yfinance as yf
        data = yf.download(
            tickers, period="2d", interval="1d",
            auto_adjust=True, progress=False, threads=True
        )
        result = {}
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    close_col = data["Close"]
                else:
                    close_col = data["Close"][ticker]
                close_col = close_col.dropna()
                if len(close_col) >= 2:
                    price = float(close_col.iloc[-1])
                    prev  = float(close_col.iloc[-2])
                    change_pct = (price - prev) / prev if prev else 0
                elif len(close_col) == 1:
                    price = float(close_col.iloc[-1])
                    change_pct = 0.0
                else:
                    continue
                result[ticker] = {"price": round(price, 2), "change_pct": round(change_pct, 4)}
            except Exception:
                continue
        return result
    except Exception as e:
        return {}

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
                result[s.ticker] = {
                    "rating": latest.rating,
                    "score": latest.weighted_score,   # use dedicated column, not committee_verdict_json
                    "date": latest.analysis_date,
                    "horizon": latest.horizon,
                }
        return result

@app.get("/api/journal/stats")
def get_journal_stats():
    """Get journal statistics from available data."""
    with get_session() as session:
        total = session.query(PredictionORM).count()
        if total == 0:
            return {
                "total_predictions": 0,
                "avg_score": None,
                "avg_confidence": None,
                "buy_rate": None,
            }
        from sqlalchemy import func
        avg_score = session.query(func.avg(PredictionORM.weighted_score)).scalar()
        avg_conf  = session.query(func.avg(PredictionORM.confidence)).scalar()
        buy_count = session.query(PredictionORM).filter(
            PredictionORM.rating.in_(["BUY", "STRONG_BUY", "STRONG BUY"])
        ).count()
        return {
            "total_predictions": total,
            "avg_score":      round(avg_score, 2) if avg_score is not None else None,
            "avg_confidence": round(avg_conf * 100, 1) if avg_conf is not None else None,
            "buy_rate":       round(buy_count / total * 100, 1) if total > 0 else None,
        }

@app.get("/api/journal/predictions")
def get_predictions():
    """Get history of all predictions"""
    with get_session() as session:
        preds = session.query(PredictionORM).order_by(PredictionORM.created_at.desc()).limit(200).all()
        result = []
        for p in preds:
            result.append({
                "id": p.prediction_id,
                "seq": p.sequence_number,
                "ticker": p.ticker,
                "date": p.analysis_date,
                "horizon": p.horizon,
                "rating": p.rating,
                "confidence": p.confidence,
                "weighted_score": p.weighted_score,
                "expected_return": p.expected_return,
            })
        return result

@app.delete("/api/journal/clear")
def clear_journal():
    """Delete all predictions from the journal (irreversible)."""
    with get_session() as session:
        count = session.query(PredictionORM).count()
        session.query(PredictionORM).delete()
        return {"status": "cleared", "deleted": count}

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
        
        # Return the format expected by renderReport
        # weighted_score: prefer the dedicated column (new), then fall back to committee_verdict_json
        score = p.weighted_score
        if score is None and committee:
            score = committee.get("weighted_score")
        # Final fallback: map rating label to approximate score for old entries
        if score is None and p.rating:
            score = {"STRONG_BUY": 9.0, "STRONG BUY": 9.0,
                     "BUY": 7.5, "HOLD": 5.0, "SELL": 2.5,
                     "STRONG_SELL": 1.0, "STRONG SELL": 1.0}.get(str(p.rating).upper())

        return {
            "ticker": p.ticker,
            "horizon": p.horizon,
            "analysis_date": p.analysis_date,
            "weighted_score": score,
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


# ── Screener endpoints ─────────────────────────────────────────────────────────

@app.get("/api/screener/indices")
def get_screener_indices():
    """Return available index options for the screener."""
    return [
        {"id": idx_id, "name": name, "size": size}
        for idx_id, (name, _, size) in INDEX_REGISTRY.items()
    ]


class ScreenerRequest(BaseModel):
    index: str = "nasdaq100"
    top_n: int = 20
    horizon: str = "12M"


@app.post("/api/screener/run")
def run_screener(req: ScreenerRequest):
    """
    Stream screener results as Server-Sent Events (SSE).
    Each event is a JSON line prefixed with 'data: '.
    """
    horizon_map = {
        "3M":   Horizon.THREE_MONTHS,
        "12M":  Horizon.TWELVE_MONTHS,
        "3-5Y": Horizon.THREE_FIVE_YEARS,
    }
    horizon = horizon_map.get(req.horizon, Horizon.TWELVE_MONTHS)

    try:
        tickers = get_index_tickers(req.index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    engine = ScreenerEngine(max_workers=12)

    def event_stream():
        for event in engine.scan_stream(tickers, top_n=req.top_n, horizon=horizon):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
