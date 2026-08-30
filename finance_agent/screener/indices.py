"""
Index constituent lists for the Market Screener.

S&P 500 and NASDAQ 100 are fetched live from Wikipedia.
DAX 40 is kept as a hardcoded fallback (stable list).
"""

from __future__ import annotations

import io
import requests
import pandas as pd
from finance_agent.utils.logger import logger

def _fetch_wiki_tickers(url: str, default_col: str = "Symbol") -> list[str]:
    """Helper to fetch tickers from Wikipedia tables safely with a custom User-Agent."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        html = requests.get(url, headers=headers, timeout=10).text
        tables = pd.read_html(io.StringIO(html), header=0)
        df = tables[0]
        col = default_col if default_col in df.columns else "Ticker"
        tickers = df[col].dropna().tolist()
        # Clean: remove any random spaces or weird chars, convert dots to dashes for yfinance
        return [str(t).strip().replace(".", "-") for t in tickers]
    except Exception as e:
        logger.error(f"[Indices] Failed fetching from {url}: {e}")
        return []


def get_sp500() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tickers = _fetch_wiki_tickers(url)
    if tickers:
        logger.info(f"[Indices] Loaded S&P 500: {len(tickers)} tickers from Wikipedia")
        return tickers
    else:
        # Minimal fallback — top 30 by market cap
        return [
            "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","BRK-B","LLY",
            "JPM","V","UNH","XOM","MA","ORCL","COST","JNJ","PG","HD",
            "WMT","MRK","ABBV","BAC","CRM","CVX","MCD","AMD","NFLX","ADBE"
        ]


def get_nasdaq100() -> list[str]:
    """Fetch NASDAQ-100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        html = requests.get(url, headers=headers, timeout=10).text
        tables = pd.read_html(io.StringIO(html), header=0)
        # Find the table that has a 'Ticker' or 'Symbol' column
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if "ticker" in cols or "symbol" in cols:
                col = "Ticker" if "Ticker" in df.columns else "Symbol"
                tickers = [str(t).strip().replace(".", "-") for t in df[col].dropna().tolist()]
                logger.info(f"[Indices] Loaded NASDAQ-100: {len(tickers)} tickers from Wikipedia")
                return tickers[:100]
        raise ValueError("No ticker column found")
    except Exception as e:
        logger.error(f"[Indices] Failed to fetch NASDAQ-100 from Wikipedia: {e}")
        return [
            "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","GOOG","AVGO","COST",
            "NFLX","TMUS","ASML","AMD","PEP","CSCO","ADBE","INTU","TXN","QCOM",
            "AMAT","ISRG","CMCSA","BKNG","HON","LRCX","VRTX","PANW","KLAC","SNPS",
            "MU","REGN","CDNS","CRWD","ADI","MELI","KDP","ABNB","MDLZ","CEG",
            "FTNT","CSGP","FAST","BIIB","ROST","AZN","MNST","DXCM","ROP","CPRT"
        ]


def get_dax40() -> list[str]:
    """DAX 40 constituents (stable — updated as needed)."""
    return [
        "ADS.DE","AIR.DE","ALV.DE","BAS.DE","BAYN.DE","BMW.DE","BNR.DE","CON.DE",
        "1COV.DE","DTG.DE","DBK.DE","DB1.DE","DHL.DE","DTE.DE","EOAN.DE","FRE.DE",
        "FME.DE","HNR1.DE","HEI.DE","HEN3.DE","IFX.DE","INL.DE","MBG.DE","MRK.DE",
        "MTX.DE","MUV2.DE","NFH.DE","PAH3.DE","P911.DE","PUM.DE","QIA.DE","RHM.DE",
        "RWE.DE","SAP.DE","SHL.DE","SIE.DE","SY1.DE","VNA.DE","VOW3.DE","ZAL.DE"
    ]


def get_sp1500() -> list[str]:
    """
    S&P 1500 Composite index (Broad US Market).
    Aggregates S&P 500, S&P 400 (MidCap), and S&P 600 (SmallCap).
    """
    sp500 = get_sp500()
    sp400 = _fetch_wiki_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
    sp600 = _fetch_wiki_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
    
    combined = sorted(list(set(sp500 + sp400 + sp600)))
    logger.info(f"[Indices] Loaded S&P 1500 Composite: {len(combined)} tickers")
    return combined


# Registry: index_id → (display_name, getter_function, approx_size)
INDEX_REGISTRY: dict[str, tuple[str, callable, int]] = {
    "sp500":    ("S&P 500",    get_sp500,    504),
    "nasdaq100":("NASDAQ 100", get_nasdaq100, 100),
    "sp1500":   ("S&P 1500 (Broad US)", get_sp1500, 1500),
    "dax40":   ("DAX 40",    get_dax40,    40),
}


def get_index_tickers(index_id: str) -> list[str]:
    """Return ticker list for a given index ID."""
    if index_id not in INDEX_REGISTRY:
        raise ValueError(f"Unknown index: {index_id}. Available: {list(INDEX_REGISTRY.keys())}")
    _, getter, _ = INDEX_REGISTRY[index_id]
    return getter()
