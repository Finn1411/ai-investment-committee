"""
Index constituent lists for the Market Screener.

S&P 500 and NASDAQ 100 are fetched live from Wikipedia.
DAX 40 is kept as a hardcoded fallback (stable list).
"""

from __future__ import annotations

import pandas as pd
from finance_agent.utils.logger import logger


def get_sp500() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, header=0)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        # yfinance uses dashes not dots (BRK.B → BRK-B)
        tickers = [t.replace(".", "-") for t in tickers]
        logger.info(f"[Indices] Loaded S&P 500: {len(tickers)} tickers from Wikipedia")
        return tickers
    except Exception as e:
        logger.error(f"[Indices] Failed to fetch S&P 500 from Wikipedia: {e}")
        # Minimal fallback — top 30 by market cap
        return [
            "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","BRK-B","LLY",
            "JPM","V","UNH","XOM","MA","ORCL","COST","JNJ","PG","HD",
            "WMT","MRK","ABBV","BAC","CRM","CVX","MCD","AMD","NFLX","ADBE"
        ]


def get_nasdaq100() -> list[str]:
    """Fetch NASDAQ-100 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url, header=0)
        # Find the table that has a 'Ticker' or 'Symbol' column
        for df in tables:
            cols = [c.lower() for c in df.columns]
            if "ticker" in cols or "symbol" in cols:
                col = "Ticker" if "Ticker" in df.columns else "Symbol"
                tickers = [t.replace(".", "-") for t in df[col].dropna().tolist()]
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


# Registry: index_id → (display_name, getter_function, approx_size)
INDEX_REGISTRY: dict[str, tuple[str, callable, int]] = {
    "sp500":    ("S&P 500",    get_sp500,    504),
    "nasdaq100":("NASDAQ 100", get_nasdaq100, 100),
    "dax40":   ("DAX 40",    get_dax40,    40),
}


def get_index_tickers(index_id: str) -> list[str]:
    """Return ticker list for a given index ID."""
    if index_id not in INDEX_REGISTRY:
        raise ValueError(f"Unknown index: {index_id}. Available: {list(INDEX_REGISTRY.keys())}")
    _, getter, _ = INDEX_REGISTRY[index_id]
    return getter()
