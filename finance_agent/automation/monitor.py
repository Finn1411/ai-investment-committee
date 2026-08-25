from finance_agent.database.db import get_session, StockORM, PredictionORM
from finance_agent.data.pipeline import DataPipeline
from finance_agent.agents.committee import CommitteeEngine
from finance_agent.models.schemas import PortfolioContext
from finance_agent.automation.alerts import AlertManager
from finance_agent.utils.logger import logger
from sqlalchemy import desc

class WatchlistMonitor:
    def __init__(self):
        self.alert_manager = AlertManager()
        self.pipeline = DataPipeline(persist_to_db=True)
        self.portfolio = PortfolioContext()
        self.committee = CommitteeEngine(portfolio=self.portfolio, persist_to_db=True, log_to_journal=True)
        
    def run_daily_scan(self):
        logger.info("[WatchlistMonitor] Starting daily watchlist scan...")
        with get_session() as session:
            # 1. Get all unique tickers we've analyzed before
            stocks = session.query(StockORM).all()
            if not stocks:
                logger.info("[WatchlistMonitor] No stocks in database. Exiting.")
                return
                
            for stock in stocks:
                ticker = stock.ticker
                logger.info(f"[WatchlistMonitor] Analyzing {ticker}")
                
                # Fetch last known score
                last_entry = session.query(PredictionORM).filter_by(ticker=ticker).order_by(desc(PredictionORM.timestamp)).first()
                last_score = last_entry.committee_score if last_entry else None
                last_rating = last_entry.rating if last_entry else None
                
                try:
                    # Run full analysis
                    context = self.pipeline.run(ticker)
                    result = self.committee.run(context, horizon=self.pipeline.horizon)
                    
                    new_score = result.final_score
                    new_rating = result.verdict
                    
                    # 2. Check for alerts
                    if last_score is not None:
                        score_diff = new_score - last_score
                        
                        # Trigger alert if rating dropped, or if score dropped by more than 1 point
                        if last_rating != new_rating and new_rating == "AVOID":
                            self.alert_manager.send_desktop_notification(
                                title=f"🚨 Rating Downgrade: {ticker}",
                                message=f"{ticker} dropped from {last_rating} to AVOID. Score: {new_score:.1f}/10"
                            )
                        elif score_diff <= -1.0:
                            self.alert_manager.send_desktop_notification(
                                title=f"⚠️ Major Score Drop: {ticker}",
                                message=f"Score fell from {last_score:.1f} to {new_score:.1f}. Rating: {new_rating}"
                            )
                        elif last_rating != new_rating and new_rating == "BUY":
                             self.alert_manager.send_desktop_notification(
                                title=f"🚀 Rating Upgrade: {ticker}",
                                message=f"{ticker} upgraded from {last_rating} to BUY. Score: {new_score:.1f}/10"
                            )
                            
                except Exception as e:
                    logger.error(f"[WatchlistMonitor] Failed to analyze {ticker}: {e}")

if __name__ == "__main__":
    monitor = WatchlistMonitor()
    monitor.run_daily_scan()
