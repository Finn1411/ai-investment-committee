import yfinance as yf
from dataclasses import dataclass
from finance_agent.utils.logger import logger

@dataclass
class MacroRegime:
    vix_level: float
    treasury_10y_yield: float
    risk_environment: str # "RISK ON", "NEUTRAL", "RISK OFF"

class MacroEngine:
    """Fetches macro indicators and determines market regime."""
    
    def fetch_regime(self) -> MacroRegime:
        try:
            logger.info("[MacroEngine] Fetching VIX and 10Y Treasury Yields...")
            vix_ticker = yf.Ticker("^VIX")
            tnx_ticker = yf.Ticker("^TNX")
            
            # Fetch latest price
            vix_hist = vix_ticker.history(period="5d")
            tnx_hist = tnx_ticker.history(period="5d")
            
            vix_level = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 20.0
            tnx_level = tnx_hist['Close'].iloc[-1] if not tnx_hist.empty else 4.0
            
            # Determine risk environment
            if vix_level > 25.0:
                env = "RISK OFF (High Volatility)"
            elif vix_level < 15.0:
                env = "RISK ON (Low Volatility)"
            else:
                env = "NEUTRAL"
                
            regime = MacroRegime(
                vix_level=vix_level,
                treasury_10y_yield=tnx_level,
                risk_environment=env
            )
            logger.info(f"[MacroEngine] Regime detected: {env} (VIX: {vix_level:.2f})")
            return regime
            
        except Exception as e:
            logger.warning(f"[MacroEngine] Failed to fetch macro data: {e}. Defaulting to Neutral.")
            return MacroRegime(vix_level=20.0, treasury_10y_yield=4.0, risk_environment="NEUTRAL")
