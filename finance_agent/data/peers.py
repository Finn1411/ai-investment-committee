import yfinance as yf
from finance_agent.utils.logger import logger

# Major Tech/Growth Peers manually mapped
PEER_MAP = {
    "MSFT": ["AAPL", "GOOGL", "AMZN"],
    "AAPL": ["MSFT", "GOOGL", "AMZN"],
    "GOOGL": ["MSFT", "META", "AMZN"],
    "AMZN": ["WMT", "MSFT", "GOOGL"],
    "NVDA": ["AMD", "INTC", "TSM"],
    "AMD": ["NVDA", "INTC"],
    "TSLA": ["F", "GM", "RIVN"],
    "PLTR": ["CRM", "SNOW", "MDB"],
    "META": ["GOOGL", "SNAP", "PINS"],
}

class PeerEngine:
    """Fetches relative peer performance."""
    
    def fetch_peer_context(self, ticker: str) -> str:
        """Returns a string describing how the stock has performed vs peers over 1Y."""
        peers = PEER_MAP.get(ticker, [])
        if not peers:
            # Fallback to S&P 500 if no peers mapped
            peers = ["^GSPC"]
            
        try:
            logger.info(f"[PeerEngine] Fetching peer performance for {ticker} vs {peers}")
            main_ticker = yf.Ticker(ticker)
            main_hist = main_ticker.history(period="1y")
            if main_hist.empty:
                return "No price history available."
                
            main_ret = (main_hist['Close'].iloc[-1] / main_hist['Close'].iloc[0]) - 1
            
            context = f"Peer Comparison (1Y Return):\n- {ticker}: {main_ret*100:.1f}%\n"
            
            for p in peers:
                p_ticker = yf.Ticker(p)
                p_hist = p_ticker.history(period="1y")
                if not p_hist.empty:
                    p_ret = (p_hist['Close'].iloc[-1] / p_hist['Close'].iloc[0]) - 1
                    context += f"- {p}: {p_ret*100:.1f}%\n"
                
            return context
            
        except Exception as e:
            logger.warning(f"[PeerEngine] Failed to fetch peers: {e}")
            return "Peer data unavailable."
