"""
SQLAlchemy ORM models and database initialisation.
Uses SQLite by default (upgradeable to PostgreSQL via DATABASE_URL in .env).
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from finance_agent.utils.config import settings
from finance_agent.utils.logger import logger


# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_engine(
    settings.database.url,
    echo=settings.database.echo,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database.url else {},
)

# Enable WAL mode for SQLite (better concurrency)
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _):
    if "sqlite" in settings.database.url:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine, autocommit=False, autoflush=False
)


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class StockORM(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(100))
    country = Column(String(50))
    currency = Column(String(10))
    exchange = Column(String(50))
    added_at = Column(DateTime, default=datetime.utcnow)


class MarketDataORM(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(String(10), nullable=False, index=True)  # ISO date string
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)


class FundamentalsORM(Base):
    __tablename__ = "fundamentals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    fiscal_year = Column(Integer, nullable=False)
    fiscal_quarter = Column(Integer)
    report_date = Column(String(10))

    gross_margin = Column(Float)
    operating_margin = Column(Float)
    net_margin = Column(Float)
    roic = Column(Float)
    roe = Column(Float)
    free_cash_flow = Column(Float)
    fcf_margin = Column(Float)
    net_debt_to_ebitda = Column(Float)
    interest_coverage = Column(Float)
    current_ratio = Column(Float)
    revenue = Column(Float)
    ebitda = Column(Float)
    net_income = Column(Float)
    eps_diluted = Column(Float)
    revenue_growth_yoy = Column(Float)
    eps_growth_yoy = Column(Float)


class ValuationORM(Base):
    __tablename__ = "valuations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    as_of = Column(String(10), nullable=False)
    pe_ratio = Column(Float)
    forward_pe = Column(Float)
    ev_to_ebit = Column(Float)
    ev_to_ebitda = Column(Float)
    price_to_sales = Column(Float)
    price_to_book = Column(Float)
    fcf_yield = Column(Float)
    enterprise_value = Column(Float)
    market_cap = Column(Float)


class PredictionORM(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(36), unique=True, nullable=False, index=True)
    sequence_number = Column(Integer, unique=True)
    ticker = Column(String(20), nullable=False, index=True)
    analysis_date = Column(String(10), nullable=False)
    horizon = Column(String(10), nullable=False)
    rating = Column(String(20), nullable=False)
    expected_return = Column(Float)
    weighted_score = Column(Float, nullable=True)   # Committee composite 0-10
    confidence = Column(Float)
    thesis = Column(Text)
    key_risks = Column(Text)             # JSON list
    invalidation_criteria = Column(Text) # JSON list
    scenario_model_json = Column(Text)   # Full ScenarioModel JSON
    committee_verdict_json = Column(Text)
    review_status = Column(String(30), default="PENDING")
    actual_return = Column(Float, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionReviewORM(Base):
    __tablename__ = "prediction_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(36), nullable=False, index=True)
    review_date = Column(String(10), nullable=False)
    actual_return = Column(Float)
    benchmark_return = Column(Float)
    alpha = Column(Float)
    rating_correct = Column(Boolean)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class AgentAnalysisORM(Base):
    __tablename__ = "agent_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(36), index=True)
    agent_name = Column(String(100), nullable=False)
    ticker = Column(String(20), nullable=False, index=True)
    analysis_date = Column(String(10), nullable=False)
    summary = Column(Text)
    key_findings = Column(Text)   # JSON list
    score = Column(Float)
    confidence = Column(Float)
    raw_output = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialised at: {settings.database.url}")


from contextlib import contextmanager

@contextmanager
def get_session():
    """Context-managed session factory with automatic commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "engine",
    "SessionLocal",
    "Base",
    "init_db",
    "get_session",
    "StockORM",
    "MarketDataORM",
    "FundamentalsORM",
    "ValuationORM",
    "PredictionORM",
    "PredictionReviewORM",
    "AgentAnalysisORM",
]
