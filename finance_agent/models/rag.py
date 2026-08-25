from enum import Enum
from pydantic import BaseModel, Field

class SourceTier(str, Enum):
    TIER_1_PRIMARY = "TIER_1_PRIMARY"
    TIER_2_SECONDARY = "TIER_2_SECONDARY"
    TIER_3_OPINION = "TIER_3_OPINION"

class StructuredClaim(BaseModel):
    """A verifiable factual claim extracted from a source text."""
    claim: str = Field(..., description="The concise, self-contained factual claim (e.g. 'Apple reported $100B in revenue').")
    source_title: str = Field(..., description="The title of the source document or article.")
    source_url: str = Field(default="", description="The URL or identifier of the source.")
    date: str = Field(..., description="The date the claim was published (YYYY-MM-DD).")
    tier: SourceTier = Field(..., description="The tier of the source (1 = Primary, 2 = Secondary, 3 = Opinion).")
    confidence_score: float = Field(..., description="The LLM's confidence that this claim is factually represented (0.0 to 1.0).")
    ticker: str = Field(..., description="The stock ticker this claim is primarily related to.")

class ClaimExtractionResult(BaseModel):
    """Result from the LLM extraction step."""
    claims: list[StructuredClaim]
