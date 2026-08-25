import chromadb
from pathlib import Path
from typing import List, Optional
import uuid

from finance_agent.models.rag import StructuredClaim
from finance_agent.rag.embeddings import GeminiEmbeddingFunction
from finance_agent.utils.logger import logger

class RAGEngine:
    def __init__(self, persist_dir: str = ".chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedding_fn = GeminiEmbeddingFunction()
        
        # Collection for storing verified claims
        self.collection = self.client.get_or_create_collection(
            name="finance_claims",
            embedding_function=self.embedding_fn
        )
        logger.info(f"[RAG] Engine initialized with DB at {self.persist_dir}")

    def upsert_claims(self, claims: List[StructuredClaim]) -> None:
        if not claims:
            return
            
        ids = []
        documents = []
        metadatas = []
        
        for claim in claims:
            # We use a unique ID for each claim
            claim_id = str(uuid.uuid4())
            ids.append(claim_id)
            
            # The document itself is the textual claim
            documents.append(claim.claim)
            
            # We store the structured data as metadata for filtering
            metadatas.append({
                "source_title": claim.source_title,
                "source_url": claim.source_url,
                "date": claim.date,
                "tier": claim.tier.value,
                "confidence_score": claim.confidence_score,
                "ticker": claim.ticker
            })
            
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"[RAG] Upserted {len(claims)} claims to ChromaDB.")

    def query_claims(
        self, 
        query: str, 
        ticker: Optional[str] = None, 
        n_results: int = 5
    ) -> List[dict]:
        """
        Query the RAG database for claims matching the semantic query.
        Returns a list of dicts with 'claim' and metadata.
        """
        where = None
        if ticker:
            where = {"ticker": ticker.upper()}
            
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        # Format results
        claims_out = []
        if results['documents'] and len(results['documents']) > 0:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            dists = results['distances'][0]
            
            for doc, meta, dist in zip(docs, metas, dists):
                claims_out.append({
                    "claim": doc,
                    "metadata": meta,
                    "distance": dist
                })
                
        return claims_out
