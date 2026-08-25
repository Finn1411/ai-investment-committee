from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
import os
from finance_agent.utils.config import settings

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "gemini-embedding-2"):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "")) 

    def __call__(self, input: Documents) -> Embeddings:
        # ChromaDB passes a list of strings
        # google-genai expects to call embed_content
        
        # Batch embed if possible, or do sequentially (GenAI usually handles list inputs)
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=input
        )
        
        # result.embeddings is a list of objects that have a .values list
        embeddings = [emb.values for emb in result.embeddings]
        return embeddings
