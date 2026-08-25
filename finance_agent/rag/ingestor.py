import yfinance as yf
from datetime import datetime
from finance_agent.rag.extractor import ClaimExtractor
from finance_agent.rag.engine import RAGEngine
from finance_agent.models.rag import SourceTier
from finance_agent.utils.logger import logger

class NewsIngestor:
    def __init__(self):
        self.extractor = ClaimExtractor()
        self.engine = RAGEngine()
        
    def ingest_ticker_news(self, ticker: str):
        """Fetch recent news from yfinance, extract claims, and store them."""
        logger.info(f"[Ingestor] Fetching news for {ticker}...")
        try:
            tick = yf.Ticker(ticker)
            news = tick.news
        except Exception as e:
            logger.error(f"[Ingestor] Failed to fetch news for {ticker}: {e}")
            return
            
        if not news:
            logger.info(f"[Ingestor] No news found for {ticker}")
            return
            
        # Limit to top 3 articles to save API limits during this demo
        for article in news[:3]:
            title = article.get("title", "")
            publisher = article.get("publisher", "Yahoo Finance")
            link = article.get("link", "")
            
            # yfinance doesn't provide the full body, but often provides a summary or first paragraph
            summary = article.get("summary", "")
            if not summary:
                # If no summary, try 'providerPublishTime' to check if we can at least get title
                summary = title
                
            pub_time = article.get("providerPublishTime")
            if pub_time:
                date_str = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d")
            else:
                date_str = datetime.utcnow().strftime("%Y-%m-%d")
                
            # We treat Yahoo Finance news as TIER 2
            tier = SourceTier.TIER_2_SECONDARY
            
            # Extract
            result = self.extractor.extract_claims(
                text=summary,
                source_title=f"{publisher}: {title}",
                source_url=link,
                date=date_str,
                tier=tier.value,
                ticker=ticker
            )
            
            # Upsert
            if result.claims:
                self.engine.upsert_claims(result.claims)
                
        logger.info(f"[Ingestor] Finished ingesting news for {ticker}")
