"""
Query Expander - Generates semantic variations and decomposes complex queries
============================================================================
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class RAGConfig:
    """Minimal config needed by expander"""
    use_mock_llm: bool = True
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "documents"
    top_k: int = 5
    rerank_top_k: int = 3
    colqwen_endpoint: str = ""
    llm_endpoint: str = ""


class QueryExpander:
    """
    Expands queries to improve retrieval coverage.
    
    Strategies:
    1. Semantic variations (synonyms, rephrasings)
    2. Sub-query decomposition for complex questions
    3. Hop count estimation for HRPO grouping
    """
    
    # Keywords indicating complex multi-hop queries
    MULTI_HOP_INDICATORS = [
        "compare", "vs", "versus", "difference", "between",
        "how does", "why does", "what happens when",
        "relationship", "impact", "affect", "influence",
        "step by step", "process", "explain how",
    ]
    
    # Common domain terms and their variations
    DOMAIN_EXPANSIONS = {
        "retrieval": ["search", "information retrieval", "document retrieval"],
        "embedding": ["vector", "representation", "encoding"],
        "llm": ["large language model", "language model", "gpt", "transformer model"],
        "rag": ["retrieval augmented generation", "retrieval-augmented"],
        "colbert": ["late interaction", "token-level retrieval"],
        "reranking": ["reranker", "re-ranking", "second stage ranking"],
        "transformer": ["attention model", "self-attention"],
    }
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
    
    def expand(self, query: str) -> dict:
        """
        Expand a query into variations and analyze complexity.
        
        Returns:
            {
                "original": str,
                "expansions": list[str],
                "sub_queries": list[str],
                "hop_count": int
            }
        """
        query_lower = query.lower()
        
        # Determine hop count (complexity)
        hop_count = self._estimate_hop_count(query_lower)
        
        # Generate expansions
        expansions = self._generate_expansions(query)
        
        # Decompose into sub-queries if complex
        sub_queries = []
        if hop_count >= 2:
            sub_queries = self._decompose_query(query)
        
        return {
            "original": query,
            "expansions": expansions,
            "sub_queries": sub_queries,
            "hop_count": hop_count,
        }
    
    def _estimate_hop_count(self, query: str) -> int:
        """
        Estimate query complexity (hop count) for HRPO grouping.

        1-hop: Simple factual queries ("What is X?", "Define Y")
        2-hop: Queries requiring synthesis, comparison, or explanation
        3+-hop: Complex multi-step reasoning with multiple aspects

        Based on Dr. Zero paper's HRPO (Hop-Grouped Relative Policy Optimization).
        """
        query_lower = query.lower().strip()

        # Count complexity indicators (weighted)
        indicator_count = sum(
            1 for indicator in self.MULTI_HOP_INDICATORS
            if indicator in query_lower
        )

        # Multiple questions indicate multi-hop (but single ? doesn't add complexity)
        question_count = query.count("?")
        extra_questions = max(0, question_count - 1)  # Only count additional questions

        # Count "and" conjunctions joining distinct concepts
        and_count = query_lower.count(" and ")

        # Check for simple definitional patterns (these are always 1-hop)
        simple_patterns = [
            query_lower.startswith("what is "),
            query_lower.startswith("what are "),
            query_lower.startswith("define "),
            query_lower.startswith("who is "),
            query_lower.startswith("who was "),
        ]

        # If it's a simple definitional query with no complexity indicators, it's 1-hop
        if any(simple_patterns) and indicator_count == 0 and and_count == 0:
            return 1

        # Calculate complexity score
        complexity_score = indicator_count + extra_questions + and_count

        if complexity_score == 0:
            return 1  # Simple query
        elif complexity_score <= 2:
            return 2  # Moderate complexity
        else:
            return 3  # High complexity
    
    def _generate_expansions(self, query: str) -> list[str]:
        """Generate semantic variations of the query."""
        expansions = [query]  # Always include original
        query_lower = query.lower()
        
        # Add domain-specific expansions
        for term, variations in self.DOMAIN_EXPANSIONS.items():
            if term in query_lower:
                for variation in variations[:2]:  # Limit variations
                    expanded = re.sub(
                        re.escape(term), 
                        variation, 
                        query, 
                        flags=re.IGNORECASE
                    )
                    if expanded != query and expanded not in expansions:
                        expansions.append(expanded)
        
        # Add question form if not already a question
        if not query.strip().endswith("?"):
            question_form = f"What is {query}?"
            expansions.append(question_form)
        
        # Limit total expansions
        return expansions[:4]
    
    def _decompose_query(self, query: str) -> list[str]:
        """
        Decompose complex query into sub-queries.
        For production, use an LLM for better decomposition.
        """
        sub_queries = []
        query_lower = query.lower()
        
        # Handle comparisons
        if any(x in query_lower for x in ["compare", "vs", "versus", "difference"]):
            # Try to extract the two things being compared
            patterns = [
                r"compare\s+(.+?)\s+(?:and|vs|versus|with)\s+(.+?)(?:\?|$)",
                r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
                r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    thing1, thing2 = match.groups()
                    sub_queries.append(f"What is {thing1.strip()}?")
                    sub_queries.append(f"What is {thing2.strip()}?")
                    break
        
        # Handle "how does X work" by asking about components
        if "how does" in query_lower or "how do" in query_lower:
            # Extract the subject
            match = re.search(r"how do(?:es)?\s+(.+?)\s+work", query_lower)
            if match:
                subject = match.group(1)
                sub_queries.append(f"What is {subject}?")
                sub_queries.append(f"What are the components of {subject}?")
        
        # Handle "and" conjunctions
        if " and " in query_lower and not sub_queries:
            parts = query.split(" and ")
            if len(parts) == 2:
                sub_queries.append(parts[0].strip() + "?")
                sub_queries.append(parts[1].strip() + "?")
        
        return sub_queries[:3]  # Limit sub-queries


# For testing
if __name__ == "__main__":
    expander = QueryExpander()
    
    test_queries = [
        "What is ColBERT?",
        "How does late interaction improve retrieval compared to dense embeddings?",
        "Compare transformer attention vs RNN gates for sequence modeling",
        "What is RAG and how does it work step by step?",
    ]
    
    for query in test_queries:
        result = expander.expand(query)
        print(f"\nQuery: {query}")
        print(f"  Hop count: {result['hop_count']}")
        print(f"  Expansions: {result['expansions']}")
        print(f"  Sub-queries: {result['sub_queries']}")
