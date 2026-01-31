"""
RAG Pipeline - Works with OpenSearch or fully mocked for demo
=============================================================
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import time
import hashlib
import numpy as np


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "documents"
    
    top_k: int = 5
    rerank_top_k: int = 3
    
    use_mock_llm: bool = True
    use_mock_embeddings: bool = True
    
    # Real endpoint names (when not mocking)
    colqwen_endpoint: str = "colqwen2-embeddings"
    llm_endpoint: str = "qwen25-vl"


class MockEmbedder:
    """Mock embedder for demo - generates deterministic embeddings from text"""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
    
    def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text"""
        # Use hash to get consistent embeddings for same text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        embedding = np.random.randn(self.dim).astype(float)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """Cosine similarity"""
        a = np.array(emb1)
        b = np.array(emb2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class MockLLM:
    """Mock LLM for demo - generates contextual responses"""
    
    RESPONSE_TEMPLATES = [
        "Based on the retrieved information about {topic}, {insight}. The key points are: {points}",
        "Looking at the documents related to {topic}, I can see that {insight}. Specifically, {points}",
        "The information suggests that {topic} involves {insight}. Important aspects include: {points}",
    ]
    
    def generate(self, query: str, context_chunks: list[dict]) -> dict:
        """Generate a response based on query and context"""
        import random
        
        # Extract key terms from query
        words = query.lower().split()
        topic_words = [w for w in words if len(w) > 3 and w not in 
                      {'what', 'how', 'does', 'the', 'and', 'for', 'with', 'this', 'that'}]
        topic = " ".join(topic_words[:3]) if topic_words else "this topic"
        
        # Build response from context
        if context_chunks:
            sources = [c.get("source", "document") for c in context_chunks[:3]]
            content_preview = context_chunks[0].get("content", "")[:200]
            
            insight = f"multiple sources ({', '.join(set(sources))}) provide relevant information"
            points = f"the documents discuss {content_preview[:100]}..."
        else:
            insight = "the available information is limited"
            points = "more context would help provide a better answer"
        
        template = random.choice(self.RESPONSE_TEMPLATES)
        response_text = template.format(topic=topic, insight=insight, points=points)
        
        # Add source citations
        if context_chunks:
            response_text += "\n\n**Sources:**\n"
            for i, chunk in enumerate(context_chunks[:3], 1):
                response_text += f"- [{i}] {chunk.get('source', 'Unknown')}\n"
        
        return {
            "text": response_text,
            "model": "mock-llm-v1",
            "tokens_used": len(response_text.split()),
        }


class MockDocumentStore:
    """In-memory document store for demo"""
    
    SAMPLE_DOCUMENTS = [
        {
            "chunk_id": "doc1_chunk1",
            "content": "ColBERT (Contextualized Late Interaction over BERT) is a neural retrieval model that employs a late interaction architecture. Unlike dense retrievers that compress documents into single vectors, ColBERT retains token-level representations and computes relevance through MaxSim operations between query and document tokens.",
            "source": "ColBERT Paper Summary",
            "metadata": {"category": "retrieval", "year": 2020}
        },
        {
            "chunk_id": "doc1_chunk2", 
            "content": "The MaxSim operation in ColBERT works by computing the maximum similarity between each query token and all document tokens, then summing these maxima. This allows for fine-grained matching while still being efficient through pre-computation of document representations.",
            "source": "ColBERT Paper Summary",
            "metadata": {"category": "retrieval", "year": 2020}
        },
        {
            "chunk_id": "doc2_chunk1",
            "content": "Dense retrieval models like DPR (Dense Passage Retrieval) encode queries and documents into single dense vectors. Similarity is computed via dot product or cosine similarity. While efficient, this approach may lose fine-grained semantic information.",
            "source": "Dense Retrieval Overview",
            "metadata": {"category": "retrieval", "year": 2020}
        },
        {
            "chunk_id": "doc3_chunk1",
            "content": "Query expansion is a technique to improve retrieval by adding related terms to the original query. Modern approaches use LLMs to generate semantic variations, helping retrieve documents that use different terminology but discuss the same concepts.",
            "source": "Query Expansion Techniques",
            "metadata": {"category": "retrieval", "year": 2023}
        },
        {
            "chunk_id": "doc4_chunk1",
            "content": "RAG (Retrieval-Augmented Generation) combines retrieval systems with language models. The retriever finds relevant documents, which are then used as context for the LLM to generate informed responses. This grounds LLM outputs in factual information.",
            "source": "RAG Architecture Guide",
            "metadata": {"category": "architecture", "year": 2023}
        },
        {
            "chunk_id": "doc4_chunk2",
            "content": "Key components of a RAG system include: 1) Document chunking and indexing, 2) Query encoding, 3) Similarity search, 4) Reranking, and 5) LLM generation with retrieved context. Each component can be optimized independently.",
            "source": "RAG Architecture Guide",
            "metadata": {"category": "architecture", "year": 2023}
        },
        {
            "chunk_id": "doc5_chunk1",
            "content": "Transformers use self-attention mechanisms to process sequences. Each token attends to all other tokens, computing attention weights that determine how much each token influences the representation of others. This enables capturing long-range dependencies.",
            "source": "Transformer Architecture",
            "metadata": {"category": "models", "year": 2017}
        },
        {
            "chunk_id": "doc6_chunk1",
            "content": "BERT (Bidirectional Encoder Representations from Transformers) is pre-trained on masked language modeling and next sentence prediction. It produces contextualized embeddings where each token's representation depends on its surrounding context.",
            "source": "BERT Explained",
            "metadata": {"category": "models", "year": 2018}
        },
        {
            "chunk_id": "doc7_chunk1",
            "content": "OpenSearch is a distributed search and analytics engine. It supports vector search through k-NN plugins, enabling semantic similarity search. Hybrid search combines traditional BM25 with vector search for improved retrieval quality.",
            "source": "OpenSearch Documentation",
            "metadata": {"category": "infrastructure", "year": 2024}
        },
        {
            "chunk_id": "doc8_chunk1",
            "content": "Restate is a durable execution framework that makes distributed applications resilient. It automatically checkpoints progress and recovers from failures, making it ideal for multi-step workflows like RAG pipelines with human-in-the-loop feedback.",
            "source": "Restate Overview",
            "metadata": {"category": "infrastructure", "year": 2024}
        },
        {
            "chunk_id": "doc9_chunk1",
            "content": "Dr. Zero is a framework from Meta that enables search agents to self-evolve without training data. It uses a proposer-solver co-evolution mechanism where one model generates questions and another learns to answer them using search.",
            "source": "Dr. Zero Paper",
            "metadata": {"category": "research", "year": 2025}
        },
        {
            "chunk_id": "doc9_chunk2",
            "content": "HRPO (Hop-Grouped Relative Policy Optimization) is introduced in Dr. Zero to reduce computational costs. It groups queries by structural complexity (number of reasoning hops) and computes group-level baselines for policy optimization.",
            "source": "Dr. Zero Paper",
            "metadata": {"category": "research", "year": 2025}
        },
        {
            "chunk_id": "doc10_chunk1",
            "content": "Late interaction models like ColBERT and ColQwen offer a middle ground between dense single-vector retrieval and cross-encoder reranking. They maintain efficiency while enabling token-level matching for better semantic understanding.",
            "source": "Late Interaction Survey",
            "metadata": {"category": "retrieval", "year": 2024}
        },
        {
            "chunk_id": "doc11_chunk1",
            "content": "Embedding models convert text into dense vector representations. Modern multimodal models like ColQwen can embed both text and images, enabling unified retrieval across modalities. The quality of embeddings directly impacts retrieval performance.",
            "source": "Embedding Models Guide",
            "metadata": {"category": "models", "year": 2024}
        },
        {
            "chunk_id": "doc12_chunk1",
            "content": "Reinforcement learning from human feedback (RLHF) improves model outputs by learning from user preferences. In search systems, user clicks, ratings, and dwell time provide signals for improving retrieval and ranking.",
            "source": "RLHF for Search",
            "metadata": {"category": "research", "year": 2023}
        },
    ]
    
    def __init__(self, embedder: MockEmbedder):
        self.embedder = embedder
        self.documents = []
        self._index_documents()
    
    def _index_documents(self):
        """Pre-compute embeddings for all documents"""
        for doc in self.SAMPLE_DOCUMENTS:
            doc_with_embedding = doc.copy()
            doc_with_embedding["embedding"] = self.embedder.embed(doc["content"])
            self.documents.append(doc_with_embedding)
    
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for similar documents"""
        scores = []
        for doc in self.documents:
            sim = self.embedder.similarity(query_embedding, doc["embedding"])
            scores.append((doc, sim))
        
        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in scores[:top_k]:
            results.append({
                "chunk_id": doc["chunk_id"],
                "content": doc["content"],
                "source": doc["source"],
                "score": round(score, 4),
                "metadata": doc.get("metadata", {}),
            })
        
        return results


class RAGPipeline:
    """Complete RAG pipeline with mock support"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize components (mock or real)
        self.embedder = MockEmbedder()
        self.llm = MockLLM()
        self.doc_store = MockDocumentStore(self.embedder)
        
        # Try to connect to real OpenSearch if available
        self._opensearch_client = None
        if not config.use_mock_embeddings:
            self._init_opensearch()
    
    def _init_opensearch(self):
        """Initialize OpenSearch client"""
        try:
            from opensearchpy import OpenSearch
            self._opensearch_client = OpenSearch(
                hosts=[{
                    "host": self.config.opensearch_host,
                    "port": self.config.opensearch_port
                }],
                http_compress=True,
            )
            # Test connection
            self._opensearch_client.info()
            print(f"✓ Connected to OpenSearch at {self.config.opensearch_host}")
        except Exception as e:
            print(f"⚠ OpenSearch not available, using mock: {e}")
            self._opensearch_client = None
    
    def retrieve(
        self, 
        query: str, 
        expanded_queries: Optional[list[str]] = None
    ) -> list[dict]:
        """Retrieve relevant chunks"""
        all_queries = [query] + (expanded_queries or [])
        
        all_results = []
        seen_ids = set()
        
        for q in all_queries:
            query_emb = self.embedder.embed(q)
            
            if self._opensearch_client:
                # Use real OpenSearch
                results = self._search_opensearch(query_emb, q)
            else:
                # Use mock store
                results = self.doc_store.search(query_emb, self.config.top_k)
            
            for r in results:
                if r["chunk_id"] not in seen_ids:
                    seen_ids.add(r["chunk_id"])
                    all_results.append(r)
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:self.config.rerank_top_k]
    
    def _search_opensearch(self, query_embedding: list[float], query_text: str) -> list[dict]:
        """Search using OpenSearch"""
        try:
            response = self._opensearch_client.search(
                index=self.config.opensearch_index,
                body={
                    "size": self.config.top_k,
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "knn": {
                                        "embedding": {
                                            "vector": query_embedding,
                                            "k": self.config.top_k
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "content": {"query": query_text, "boost": 0.3}
                                    }
                                }
                            ]
                        }
                    }
                }
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "chunk_id": hit["_source"]["chunk_id"],
                    "content": hit["_source"]["content"],
                    "source": hit["_source"]["source"],
                    "score": hit["_score"],
                    "metadata": hit["_source"].get("metadata", {}),
                })
            return results
        except Exception as e:
            print(f"OpenSearch error, falling back to mock: {e}")
            return self.doc_store.search(query_embedding, self.config.top_k)
    
    def generate(self, query: str, chunks: list[dict]) -> dict:
        """Generate response using LLM"""
        if self.config.use_mock_llm:
            return self.llm.generate(query, chunks)
        else:
            # TODO: Integrate real LLM (Qwen, Claude, etc.)
            return self.llm.generate(query, chunks)
