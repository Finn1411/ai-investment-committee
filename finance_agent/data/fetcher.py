"""
yfinance Data Fetcher — raw data acquisition layer.

Fetches and normalises data from Yahoo Finance into clean Python dicts/DataFrames.
No calculations happen here — pure I/O.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from finance_agent.utils.logger import logger


# ── Raw data container ────────────────────────────────────────────────────────

@dataclass
class RawTickerData:
    """All raw data fetched from yfinance for one ticker."""
    ticker: str
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    # Price history (daily OHLCV, 2y)
    price_history: Optional[pd.DataFrame] = None

    # Info dict (fundamentals + meta)
    info: dict[str, Any] = field(default_factory=dict)

    # Financial statements (annual)
    income_stmt: Optional[pd.DataFrame] = None      # rows = line items, cols = dates
    balance_sheet: Optional[pd.DataFrame] = None
    cash_flow: Optional[pd.DataFrame] = None

    # Quarterly versions
    income_stmt_q: Optional[pd.DataFrame] = None
    balance_sheet_q: Optional[pd.DataFrame] = None
    cash_flow_q: Optional[pd.DataFrame] = None

    # Earnings history
    earnings_dates: Optional[pd.DataFrame] = None

    # Fetch errors (non-fatal)
    errors: list[str] = field(default_factory=list)


class YFinanceFetcher:
    """
    Wraps yfinance to pull all data needed for a full analysis.
    Retries on transient failures, logs all errors.
    """

    def __init__(self, retries: int = 3, retry_delay: float = 2.0) -> None:
        self.retries = retries
        self.retry_delay = retry_delay

    def fetch(self, ticker: str, price_period: str = "2y") -> RawTickerData:
        """
        Fetch all data for a ticker.

        Args:
            ticker: e.g. 'AAPL', 'MSFT'
            price_period: yfinance period string, default '2y'

        Returns:
            RawTickerData with everything populated (or errors logged)
        """
        logger.info(f"[Fetcher] Starting fetch for {ticker}")
        raw = RawTickerData(ticker=ticker.upper())
        yf_ticker = yf.Ticker(ticker)

        raw.price_history = self._fetch_prices(yf_ticker, ticker, price_period)
        raw.info = self._fetch_info(yf_ticker, ticker)
        raw.income_stmt, raw.income_stmt_q = self._fetch_income(yf_ticker, ticker)
        raw.balance_sheet, raw.balance_sheet_q = self._fetch_balance(yf_ticker, ticker)
        raw.cash_flow, raw.cash_flow_q = self._fetch_cashflow(yf_ticker, ticker)
        raw.earnings_dates = self._fetch_earnings_dates(yf_ticker, ticker)

        logger.info(
            f"[Fetcher] {ticker} done — "
            f"price rows: {len(raw.price_history) if raw.price_history is not None else 0}, "
            f"errors: {len(raw.errors)}"
        )
        return raw

    # ── Private fetch helpers ─────────────────────────────────────────────────

    def _fetch_prices(
        self, yf_ticker: yf.Ticker, ticker: str, period: str
    ) -> Optional[pd.DataFrame]:
        for attempt in range(self.retries):
            try:
                df = yf_ticker.history(period=period, auto_adjust=True)
                if df is None or df.empty:
                    raise ValueError("Empty price history")
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "Date"
                logger.debug(f"[Fetcher] {ticker} prices: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"[Fetcher] {ticker} price attempt {attempt+1}: {e}")
                time.sleep(self.retry_delay)
        logger.error(f"[Fetcher] {ticker} price fetch failed after {self.retries} retries")
        return None

    def _fetch_info(self, yf_ticker: yf.Ticker, ticker: str) -> dict:
        try:
            info = yf_ticker.info or {}
            logger.debug(f"[Fetcher] {ticker} info keys: {len(info)}")
            return info
        except Exception as e:
            logger.warning(f"[Fetcher] {ticker} info fetch failed: {e}")
            return {}

    def _fetch_income(
        self, yf_ticker: yf.Ticker, ticker: str
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        annual, quarterly = None, None
        try:
            annual = yf_ticker.financials
            quarterly = yf_ticker.quarterly_financials
        except Exception as e:
            logger.warning(f"[Fetcher] {ticker} income stmt failed: {e}")
        return annual, quarterly

    def _fetch_balance(
        self, yf_ticker: yf.Ticker, ticker: str
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        annual, quarterly = None, None
        try:
            annual = yf_ticker.balance_sheet
            quarterly = yf_ticker.quarterly_balance_sheet
        except Exception as e:
            logger.warning(f"[Fetcher] {ticker} balance sheet failed: {e}")
        return annual, quarterly

    def _fetch_cashflow(
        self, yf_ticker: yf.Ticker, ticker: str
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        annual, quarterly = None, None
        try:
            annual = yf_ticker.cashflow
            quarterly = yf_ticker.quarterly_cashflow
        except Exception as e:
            logger.warning(f"[Fetcher] {ticker} cashflow failed: {e}")
        return annual, quarterly

    def _fetch_earnings_dates(
        self, yf_ticker: yf.Ticker, ticker: str
    ) -> Optional[pd.DataFrame]:
        import warnings, io, contextlib
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return yf_ticker.earnings_dates
        except Exception as e:
            logger.debug(f"[Fetcher] {ticker} earnings_dates not available: {e}")
            return None

    # ── Normalise helpers (used by pipeline) ─────────────────────────────────

    @staticmethod
    def get_statement_row(
        df: Optional[pd.DataFrame], *row_names: str
    ) -> Optional[pd.Series]:
        """Try multiple row name variants and return first match."""
        if df is None or df.empty:
            return None
        for name in row_names:
            if name in df.index:
                return df.loc[name]
        return None

    @staticmethod
    def latest_value(series: Optional[pd.Series]) -> Optional[float]:
        """Return most recent non-NaN value from a financial statement row."""
        if series is None:
            return None
        clean = series.dropna()
        if clean.empty:
            return None
        return float(clean.iloc[0])  # Most recent column is first in yfinance

    @staticmethod
    def value_n_periods_ago(
        series: Optional[pd.Series], n: int
    ) -> Optional[float]:
        """Return value n periods ago (n=1 → prior year)."""
        if series is None:
            return None
        clean = series.dropna()
        if len(clean) <= n:
            return None
        return float(clean.iloc[n])
