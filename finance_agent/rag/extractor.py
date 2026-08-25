from google import genai
from google.genai import types
from finance_agent.models.rag import ClaimExtractionResult
from finance_agent.utils.logger import logger
from finance_agent.utils.config import settings
import os

class ClaimExtractor:
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
        self.model_name = model_name
        
    def extract_claims(self, text: str, source_title: str, source_url: str, date: str, tier: str, ticker: str) -> ClaimExtractionResult:
        """Extract structured claims from raw text."""
        
        prompt = f"""
        You are a financial analyst. Extract all important factual claims from the following text related to {ticker}.
        Only extract statements that are meaningful for evaluating the company (e.g. financials, partnerships, product launches, market trends).
        Ignore fluff or generic statements.
        
        For each claim:
        1. Formulate it as a concise, self-contained factual sentence.
        2. Set the confidence score (0.0 to 1.0) based on how explicitly it is stated in the text.
        3. Use the provided metadata for source and tier.
        
        Text to analyze:
        {text}
        
        Metadata:
        - Source Title: {source_title}
        - Source URL: {source_url}
        - Date: {date}
        - Tier: {tier}
        - Ticker: {ticker}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClaimExtractionResult,
                    temperature=0.1
                )
            )
            
            if response.parsed:
                logger.info(f"[Extractor] Extracted {len(response.parsed.claims)} claims from {source_title}")
                return response.parsed
            else:
                logger.warning(f"[Extractor] Failed to parse claims from {source_title}")
                return ClaimExtractionResult(claims=[])
                
        except Exception as e:
            logger.error(f"[Extractor] Extraction failed: {e}")
            return ClaimExtractionResult(claims=[])
