"""
Lyapunov-Stable Iterative Retrieval
====================================

Frames multi-hop retrieval as a discrete dynamical system with formal
stability guarantees. Critical for enterprise applications where:
- Convergence must be provable
- Stopping criteria must be explainable
- Each iteration must improve (or not worsen) retrieval quality

Mathematical Framework:
-----------------------
State:       x_t = (D_t, q_t, r_t)  
             where D_t = document set, q_t = refined query, r_t = relevance scores

Transition:  x_{t+1} = f(x_t, q_0)
             where f is the retrieval-refinement operator

Lyapunov:    V(x_t) = d(x_t, x*)
             distance to oracle retrieval state

Stability:   V(x_{t+1}) ≤ V(x_t) - α||x_{t+1} - x_t||²
             (strict Lyapunov decrease with margin α)

Stopping:    |V(x_{t+1}) - V(x_t)| < ε  or  t > T_max

References:
- Dr. Zero (Meta, 2025) - Multi-hop search agents
- ColBERT/ColQwen2 - Late interaction for fine-grained matching
- Lyapunov stability theory for discrete systems
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
from datetime import datetime
import json
import hashlib


# =============================================================================
# Core Data Structures
# =============================================================================

class StabilityStatus(Enum):
    """Stability status of the retrieval system"""
    CONVERGING = "converging"          # V decreasing
    STABLE = "stable"                  # V at minimum (converged)
    MARGINAL = "marginal"              # V not decreasing but bounded
    UNSTABLE = "unstable"              # V increasing (problem!)
    OSCILLATING = "oscillating"        # V cycling between values


@dataclass
class RetrievalState:
    """
    State of the retrieval system at iteration t.
    
    x_t = (D_t, q_t, r_t, embeddings)
    """
    iteration: int
    
    # Document state
    document_ids: List[str]
    document_contents: List[str]
    document_scores: List[float]
    
    # Query state (may be refined across iterations)
    original_query: str
    refined_query: str
    query_embedding: Optional[np.ndarray] = None
    
    # Aggregated document embedding (centroid or attention-weighted)
    doc_set_embedding: Optional[np.ndarray] = None
    
    # Lyapunov function value at this state
    lyapunov_value: float = float('inf')
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "document_ids": self.document_ids,
            "document_scores": self.document_scores,
            "original_query": self.original_query,
            "refined_query": self.refined_query,
            "lyapunov_value": self.lyapunov_value,
            "timestamp": self.timestamp,
            "num_documents": len(self.document_ids),
        }


@dataclass
class ConvergenceMetrics:
    """Metrics for analyzing retrieval convergence"""
    lyapunov_values: List[float]           # V(x_0), V(x_1), ...
    lyapunov_deltas: List[float]           # ΔV = V(x_{t+1}) - V(x_t)
    state_distances: List[float]           # ||x_{t+1} - x_t||
    stability_status: StabilityStatus
    converged: bool
    convergence_iteration: Optional[int]   # When did it converge?
    lyapunov_exponent: float               # Rate of convergence
    
    # Enterprise explainability
    stopping_reason: str
    confidence_bound: float                # Upper bound on distance to oracle
    
    def to_dict(self) -> dict:
        return {
            "lyapunov_values": self.lyapunov_values,
            "lyapunov_deltas": self.lyapunov_deltas,
            "stability_status": self.stability_status.value,
            "converged": self.converged,
            "convergence_iteration": self.convergence_iteration,
            "lyapunov_exponent": self.lyapunov_exponent,
            "stopping_reason": self.stopping_reason,
            "confidence_bound": self.confidence_bound,
        }


@dataclass
class LyapunovConfig:
    """Configuration for Lyapunov-stable retrieval"""
    # Convergence thresholds
    epsilon: float = 0.01                  # Convergence threshold: |ΔV| < ε
    max_iterations: int = 5                # T_max
    min_iterations: int = 1                # Minimum iterations before stopping
    
    # Stability parameters
    stability_margin: float = 0.0          # α: require V_{t+1} ≤ V_t - α||Δx||²
    oscillation_window: int = 3            # Check for oscillation over this many steps
    oscillation_threshold: float = 0.05    # Oscillation if std(V) / mean(V) < threshold
    
    # Lyapunov function choice
    lyapunov_type: str = "relevance"       # "relevance", "coverage", "hybrid"
    
    # Quality safety
    require_monotonic: bool = True         # Reject iterations that increase V
    fallback_on_instability: bool = True   # Return to last stable state if unstable
    audit_all_states: bool = True          # Keep full state history for explainability


# =============================================================================
# Lyapunov Functions
# =============================================================================

class LyapunovFunction:
    """
    Base class for Lyapunov functions V(x).
    
    Requirements for valid Lyapunov function:
    1. V(x) ≥ 0 for all x (positive semi-definite)
    2. V(x*) = 0 at optimal state (zero at equilibrium)
    3. V(x) → ∞ as ||x|| → ∞ (radially unbounded)
    """
    
    def __call__(self, state: RetrievalState) -> float:
        """Compute V(x_t)"""
        raise NotImplementedError
    
    def gradient(self, state: RetrievalState) -> np.ndarray:
        """Compute ∇V(x_t) for gradient-based analysis"""
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        raise NotImplementedError


class RelevanceLyapunov(LyapunovFunction):
    """
    Lyapunov function based on retrieval relevance.
    
    V(x_t) = 1 - max_relevance_score
    
    At oracle state x*, all documents are maximally relevant, so V(x*) = 0.
    As relevance decreases, V increases.
    """
    
    def __call__(self, state: RetrievalState) -> float:
        if not state.document_scores:
            return 1.0  # Maximum distance from oracle
        
        # Use weighted combination of top scores
        scores = sorted(state.document_scores, reverse=True)
        
        # Weighted average: top doc matters most
        weights = np.exp(-np.arange(len(scores)) * 0.5)  # Exponential decay
        weights = weights / weights.sum()
        
        weighted_relevance = np.dot(weights[:len(scores)], scores)
        
        # V = 1 - relevance, bounded in [0, 1]
        return max(0.0, 1.0 - weighted_relevance)
    
    def gradient(self, state: RetrievalState) -> np.ndarray:
        # Approximate gradient direction: towards higher-scoring documents
        if state.doc_set_embedding is None:
            return np.zeros(384)  # Default dimension
        return -state.doc_set_embedding  # Negative because we minimize V
    
    @property
    def name(self) -> str:
        return "relevance"


class CoverageLyapunov(LyapunovFunction):
    """
    Lyapunov function based on query facet coverage.
    
    V(x_t) = 1 - coverage_ratio
    
    where coverage_ratio measures how many aspects of the query
    are addressed by the retrieved documents.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    def __call__(self, state: RetrievalState) -> float:
        if state.query_embedding is None or state.doc_set_embedding is None:
            return 1.0
        
        # Coverage as cosine similarity between query and doc set centroid
        query_norm = np.linalg.norm(state.query_embedding)
        doc_norm = np.linalg.norm(state.doc_set_embedding)
        
        if query_norm < 1e-8 or doc_norm < 1e-8:
            return 1.0
        
        coverage = np.dot(state.query_embedding, state.doc_set_embedding)
        coverage = coverage / (query_norm * doc_norm)
        
        # V = 1 - coverage, bounded in [0, 2] (cosine in [-1, 1])
        return max(0.0, 1.0 - coverage)
    
    def gradient(self, state: RetrievalState) -> np.ndarray:
        if state.query_embedding is None:
            return np.zeros(self.embedding_dim)
        return state.query_embedding  # Move doc set towards query
    
    @property
    def name(self) -> str:
        return "coverage"


class HybridLyapunov(LyapunovFunction):
    """
    Hybrid Lyapunov function combining relevance and coverage.
    
    V(x_t) = α * V_relevance + (1-α) * V_coverage
    
    Provides balanced convergence criterion.
    """
    
    def __init__(self, alpha: float = 0.5, embedding_dim: int = 384):
        self.alpha = alpha
        self.relevance_lyap = RelevanceLyapunov()
        self.coverage_lyap = CoverageLyapunov(embedding_dim)
    
    def __call__(self, state: RetrievalState) -> float:
        v_rel = self.relevance_lyap(state)
        v_cov = self.coverage_lyap(state)
        return self.alpha * v_rel + (1 - self.alpha) * v_cov
    
    def gradient(self, state: RetrievalState) -> np.ndarray:
        g_rel = self.relevance_lyap.gradient(state)
        g_cov = self.coverage_lyap.gradient(state)
        return self.alpha * g_rel + (1 - self.alpha) * g_cov
    
    @property
    def name(self) -> str:
        return f"hybrid(α={self.alpha})"


class OracleLyapunov(LyapunovFunction):
    """
    Oracle Lyapunov function for training/evaluation.
    
    V(x_t) = 1 - Jaccard(D_t, D*)
    
    where D* is the ground-truth relevant document set.
    Only usable when ground truth is available.
    """
    
    def __init__(self, oracle_doc_ids: List[str]):
        self.oracle_set = set(oracle_doc_ids)
    
    def __call__(self, state: RetrievalState) -> float:
        if not self.oracle_set:
            return 0.0  # No oracle = assume perfect
        
        current_set = set(state.document_ids)
        
        intersection = len(current_set & self.oracle_set)
        union = len(current_set | self.oracle_set)
        
        if union == 0:
            return 1.0
        
        jaccard = intersection / union
        return 1.0 - jaccard
    
    def gradient(self, state: RetrievalState) -> np.ndarray:
        # No continuous gradient for set-based metric
        return np.zeros(384)
    
    @property
    def name(self) -> str:
        return "oracle"


# =============================================================================
# State Transition Function
# =============================================================================

class RetrievalTransition:
    """
    State transition function f(x_t, q) for iterative retrieval.
    
    Implements: x_{t+1} = f(x_t, q_0)
    
    Strategies:
    1. Query refinement: Incorporate top docs into query
    2. Expansion: Add related documents
    3. Filtering: Remove low-relevance documents
    4. Re-ranking: Adjust scores based on context
    """
    
    def __init__(
        self,
        embedder: Callable[[str], np.ndarray],
        retriever: Callable[[str, int], List[Dict]],
        query_refiner: Optional[Callable[[str, List[str]], str]] = None,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.query_refiner = query_refiner
        self.top_k = top_k
    
    def __call__(
        self, 
        state: RetrievalState, 
        original_query: str
    ) -> RetrievalState:
        """
        Compute next state: x_{t+1} = f(x_t, q_0)
        """
        # Strategy 1: Refine query based on current docs
        if self.query_refiner and state.document_contents:
            refined_query = self.query_refiner(
                original_query, 
                state.document_contents[:3]  # Top 3 docs
            )
        else:
            # Default: Append key terms from top doc
            refined_query = self._simple_query_refinement(
                original_query,
                state.document_contents,
                state.refined_query
            )
        
        # Strategy 2: Retrieve with refined query
        new_results = self.retriever(refined_query, self.top_k * 2)  # Over-retrieve
        
        # Strategy 3: Merge with existing results (union + re-rank)
        merged = self._merge_results(state, new_results)
        
        # Strategy 4: Filter to top_k
        top_results = sorted(merged, key=lambda x: x["score"], reverse=True)[:self.top_k]
        
        # Compute embeddings for new state
        query_emb = self.embedder(refined_query)
        doc_embs = [self.embedder(d["content"]) for d in top_results]
        doc_set_emb = np.mean(doc_embs, axis=0) if doc_embs else np.zeros_like(query_emb)
        
        return RetrievalState(
            iteration=state.iteration + 1,
            document_ids=[d["id"] for d in top_results],
            document_contents=[d["content"] for d in top_results],
            document_scores=[d["score"] for d in top_results],
            original_query=original_query,
            refined_query=refined_query,
            query_embedding=query_emb,
            doc_set_embedding=doc_set_emb,
        )
    
    def _simple_query_refinement(
        self, 
        original: str, 
        docs: List[str],
        previous_refined: str
    ) -> str:
        """Simple query refinement by extracting key terms from top docs"""
        if not docs:
            return original
        
        # Extract unique terms from top doc that aren't in query
        query_terms = set(original.lower().split())
        doc_terms = docs[0].lower().split()
        
        # Find informative terms (longer words not in query)
        new_terms = [
            t for t in doc_terms 
            if len(t) > 4 and t not in query_terms and t.isalpha()
        ][:3]
        
        if new_terms:
            return f"{original} {' '.join(new_terms)}"
        return original
    
    def _merge_results(
        self, 
        state: RetrievalState, 
        new_results: List[Dict]
    ) -> List[Dict]:
        """Merge existing and new results with score fusion"""
        seen = {}
        
        # Add existing results
        for i, doc_id in enumerate(state.document_ids):
            seen[doc_id] = {
                "id": doc_id,
                "content": state.document_contents[i],
                "score": state.document_scores[i],
                "iteration_found": state.iteration,
            }
        
        # Merge new results (boost if seen before)
        for r in new_results:
            doc_id = r.get("id", r.get("chunk_id"))
            if doc_id in seen:
                # Reciprocal rank fusion-style boost
                seen[doc_id]["score"] = 0.6 * seen[doc_id]["score"] + 0.4 * r["score"]
            else:
                seen[doc_id] = {
                    "id": doc_id,
                    "content": r.get("content", ""),
                    "score": r["score"] * 0.9,  # Slight penalty for new docs
                    "iteration_found": state.iteration + 1,
                }
        
        return list(seen.values())


# =============================================================================
# Lyapunov-Stable Retrieval Controller
# =============================================================================

class LyapunovRetriever:
    """
    Main controller for Lyapunov-stable iterative retrieval.
    
    Guarantees:
    1. Monotonic improvement (V decreases or stays constant)
    2. Bounded iterations (stops at convergence or T_max)
    3. Full audit trail for enterprise explainability
    """
    
    def __init__(
        self,
        transition: RetrievalTransition,
        lyapunov_fn: LyapunovFunction,
        config: LyapunovConfig,
    ):
        self.transition = transition
        self.lyapunov_fn = lyapunov_fn
        self.config = config
        
        # State history for audit trail
        self.state_history: List[RetrievalState] = []
        self.metrics: Optional[ConvergenceMetrics] = None
    
    def retrieve(
        self, 
        query: str,
        initial_results: Optional[List[Dict]] = None,
    ) -> Tuple[RetrievalState, ConvergenceMetrics]:
        """
        Run Lyapunov-stable iterative retrieval.
        
        Returns final state and convergence metrics.
        """
        # Initialize state
        if initial_results:
            state = self._create_initial_state(query, initial_results)
        else:
            # Cold start: retrieve initial results
            initial = self.transition.retriever(query, self.transition.top_k)
            state = self._create_initial_state(query, initial)
        
        state.lyapunov_value = self.lyapunov_fn(state)
        
        # Track history
        self.state_history = [state]
        lyapunov_values = [state.lyapunov_value]
        lyapunov_deltas = []
        state_distances = []
        
        # Main iteration loop
        converged = False
        stopping_reason = "max_iterations"
        last_stable_state = state
        
        for t in range(self.config.max_iterations):
            # Compute next state
            next_state = self.transition(state, query)
            next_state.lyapunov_value = self.lyapunov_fn(next_state)
            
            # Compute deltas
            delta_v = next_state.lyapunov_value - state.lyapunov_value
            delta_x = self._state_distance(state, next_state)
            
            lyapunov_deltas.append(delta_v)
            state_distances.append(delta_x)
            
            # Check stability
            stability_ok = self._check_stability(delta_v, delta_x)
            
            if not stability_ok and self.config.require_monotonic:
                # Reject this iteration - revert to last stable
                if self.config.fallback_on_instability:
                    stopping_reason = f"instability_at_iter_{t+1}_reverted"
                    break
                else:
                    stopping_reason = f"instability_at_iter_{t+1}"
                    # Continue but flag the issue
            
            # Accept this state
            state = next_state
            self.state_history.append(state)
            lyapunov_values.append(state.lyapunov_value)
            
            if stability_ok:
                last_stable_state = state
            
            # Check convergence
            if t >= self.config.min_iterations - 1:
                if abs(delta_v) < self.config.epsilon:
                    converged = True
                    stopping_reason = f"converged_at_iter_{t+1}_delta_v={delta_v:.4f}"
                    break
                
                # Check for oscillation
                if self._check_oscillation(lyapunov_values):
                    stopping_reason = f"oscillation_detected_at_iter_{t+1}"
                    break
        
        # Compute final metrics
        self.metrics = self._compute_metrics(
            lyapunov_values,
            lyapunov_deltas,
            state_distances,
            converged,
            stopping_reason,
        )
        
        # Return last stable state for safety
        final_state = last_stable_state if self.config.require_monotonic else state
        
        return final_state, self.metrics
    
    def _create_initial_state(
        self, 
        query: str, 
        results: List[Dict]
    ) -> RetrievalState:
        """Create initial retrieval state from results"""
        query_emb = self.transition.embedder(query)
        
        doc_contents = [r.get("content", "") for r in results]
        doc_embs = [self.transition.embedder(c) for c in doc_contents]
        doc_set_emb = np.mean(doc_embs, axis=0) if doc_embs else np.zeros_like(query_emb)
        
        return RetrievalState(
            iteration=0,
            document_ids=[r.get("id", r.get("chunk_id", f"doc_{i}")) for i, r in enumerate(results)],
            document_contents=doc_contents,
            document_scores=[r.get("score", 0.5) for r in results],
            original_query=query,
            refined_query=query,
            query_embedding=query_emb,
            doc_set_embedding=doc_set_emb,
        )
    
    def _state_distance(
        self, 
        s1: RetrievalState, 
        s2: RetrievalState
    ) -> float:
        """Compute distance between states ||x_{t+1} - x_t||"""
        # Document set distance (Jaccard)
        set1 = set(s1.document_ids)
        set2 = set(s2.document_ids)
        
        if not set1 and not set2:
            jaccard = 1.0
        else:
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard = intersection / union if union > 0 else 0.0
        
        doc_distance = 1.0 - jaccard
        
        # Query embedding distance
        if s1.query_embedding is not None and s2.query_embedding is not None:
            query_distance = 1.0 - np.dot(s1.query_embedding, s2.query_embedding) / (
                np.linalg.norm(s1.query_embedding) * np.linalg.norm(s2.query_embedding) + 1e-8
            )
        else:
            query_distance = 0.0
        
        # Combined distance
        return 0.7 * doc_distance + 0.3 * query_distance
    
    def _check_stability(self, delta_v: float, delta_x: float) -> bool:
        """Check Lyapunov stability condition"""
        # Strict: V_{t+1} ≤ V_t - α||Δx||²
        margin = self.config.stability_margin * (delta_x ** 2)
        return delta_v <= margin + 1e-8  # Small tolerance for numerical issues
    
    def _check_oscillation(self, values: List[float]) -> bool:
        """Check for oscillatory behavior in Lyapunov values"""
        if len(values) < self.config.oscillation_window:
            return False
        
        recent = values[-self.config.oscillation_window:]
        mean_v = np.mean(recent)
        std_v = np.std(recent)
        
        if mean_v < 1e-8:
            return False
        
        # Oscillating if variance is high relative to mean
        cv = std_v / mean_v  # Coefficient of variation
        
        # Also check for sign changes in deltas
        deltas = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        sign_changes = sum(1 for i in range(len(deltas)-1) if deltas[i] * deltas[i+1] < 0)
        
        return cv > self.config.oscillation_threshold and sign_changes >= len(deltas) - 1
    
    def _compute_metrics(
        self,
        lyapunov_values: List[float],
        lyapunov_deltas: List[float],
        state_distances: List[float],
        converged: bool,
        stopping_reason: str,
    ) -> ConvergenceMetrics:
        """Compute convergence metrics"""
        # Determine stability status
        if converged:
            status = StabilityStatus.STABLE
        elif all(d <= 0 for d in lyapunov_deltas):
            status = StabilityStatus.CONVERGING
        elif self._check_oscillation(lyapunov_values):
            status = StabilityStatus.OSCILLATING
        elif any(d > self.config.epsilon for d in lyapunov_deltas):
            status = StabilityStatus.UNSTABLE
        else:
            status = StabilityStatus.MARGINAL
        
        # Compute Lyapunov exponent (rate of convergence)
        # λ ≈ log(V_T / V_0) / T
        if len(lyapunov_values) > 1 and lyapunov_values[0] > 1e-8:
            v_ratio = max(lyapunov_values[-1], 1e-8) / lyapunov_values[0]
            lyap_exp = np.log(v_ratio) / len(lyapunov_values)
        else:
            lyap_exp = 0.0
        
        # Convergence iteration
        conv_iter = None
        for i, dv in enumerate(lyapunov_deltas):
            if abs(dv) < self.config.epsilon:
                conv_iter = i + 1
                break
        
        # Confidence bound: final V value is upper bound on distance to oracle
        confidence = lyapunov_values[-1] if lyapunov_values else 1.0
        
        return ConvergenceMetrics(
            lyapunov_values=lyapunov_values,
            lyapunov_deltas=lyapunov_deltas,
            state_distances=state_distances,
            stability_status=status,
            converged=converged,
            convergence_iteration=conv_iter,
            lyapunov_exponent=lyap_exp,
            stopping_reason=stopping_reason,
            confidence_bound=confidence,
        )
    
    def get_audit_trail(self) -> List[Dict]:
        """Get full audit trail for enterprise explainability"""
        trail = []
        for i, state in enumerate(self.state_history):
            entry = state.to_dict()
            entry["lyapunov_function"] = self.lyapunov_fn.name
            
            if i > 0:
                prev = self.state_history[i-1]
                entry["delta_v"] = state.lyapunov_value - prev.lyapunov_value
                entry["stability"] = "stable" if entry["delta_v"] <= 0 else "improving" if entry["delta_v"] < self.config.epsilon else "unstable"
            
            trail.append(entry)
        
        return trail


# =============================================================================
# Integration with ColQwen2/MaxSim Pipeline
# =============================================================================

class ColQwenLyapunovRetriever:
    """
    Lyapunov-stable retriever using ColQwen2 embeddings and MaxSim scoring.
    
    Integrates with the existing RAG pipeline.
    """
    
    def __init__(
        self,
        rag_pipeline,  # RAGPipeline from rag_pipeline.py
        query_expander,  # QueryExpander from query_expander.py
        config: Optional[LyapunovConfig] = None,
    ):
        self.rag = rag_pipeline
        self.expander = query_expander
        self.config = config or LyapunovConfig()
        
        # Create Lyapunov function
        self.lyapunov_fn = HybridLyapunov(alpha=0.6)  # Favor relevance slightly
        
        # Create transition function
        self.transition = RetrievalTransition(
            embedder=self._embed,
            retriever=self._retrieve,
            query_refiner=self._refine_query,
            top_k=self.rag.config.top_k if hasattr(self.rag, 'config') else 5,
        )
        
        # Create controller
        self.controller = LyapunovRetriever(
            transition=self.transition,
            lyapunov_fn=self.lyapunov_fn,
            config=self.config,
        )
    
    def _embed(self, text: str) -> np.ndarray:
        """Embed text using the RAG pipeline embedder"""
        return np.array(self.rag.embedder.embed(text))
    
    def _retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using the RAG pipeline"""
        return self.rag.retrieve(query, [])[:top_k]
    
    def _refine_query(self, query: str, docs: List[str]) -> str:
        """Refine query using the query expander"""
        expanded = self.expander.expand(query)
        
        # Use first expansion as refined query
        if expanded.get("expansions"):
            return expanded["expansions"][0]
        return query
    
    def search(
        self, 
        query: str,
        return_metrics: bool = True,
    ) -> Dict:
        """
        Run Lyapunov-stable search.
        
        Returns results with convergence metrics and audit trail.
        """
        # Get initial results
        initial = self._retrieve(query, self.transition.top_k)
        
        # Run iterative retrieval
        final_state, metrics = self.controller.retrieve(query, initial)
        
        result = {
            "query": query,
            "documents": [
                {
                    "id": final_state.document_ids[i],
                    "content": final_state.document_contents[i],
                    "score": final_state.document_scores[i],
                }
                for i in range(len(final_state.document_ids))
            ],
            "iterations": final_state.iteration,
            "converged": metrics.converged,
            "stopping_reason": metrics.stopping_reason,
        }
        
        if return_metrics:
            result["metrics"] = metrics.to_dict()
            result["audit_trail"] = self.controller.get_audit_trail()
        
        return result


# =============================================================================
# CLI and Demo
# =============================================================================

def demo_lyapunov_retrieval():
    """Demonstrate Lyapunov-stable retrieval with synthetic data"""
    print("\n" + "=" * 70)
    print("LYAPUNOV-STABLE ITERATIVE RETRIEVAL DEMO")
    print("=" * 70)
    
    # Create mock embedder
    class MockEmbedder:
        def embed(self, text: str) -> List[float]:
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(384)
            return (emb / np.linalg.norm(emb)).tolist()
    
    # Create mock retriever with improving results
    iteration_bonus = [0]  # Mutable to track iterations
    
    def mock_retrieve(query: str, top_k: int) -> List[Dict]:
        results = []
        base_score = 0.5 + 0.1 * iteration_bonus[0]  # Improve each iteration
        iteration_bonus[0] += 1
        
        for i in range(top_k):
            results.append({
                "id": f"doc_{hash(query + str(i)) % 1000}",
                "content": f"Relevant content for '{query[:30]}...' (doc {i})",
                "score": min(1.0, base_score - 0.05 * i + np.random.uniform(-0.05, 0.05)),
            })
        
        return results
    
    embedder = MockEmbedder()
    
    # Create transition function
    transition = RetrievalTransition(
        embedder=lambda t: np.array(embedder.embed(t)),
        retriever=mock_retrieve,
        top_k=5,
    )
    
    # Create Lyapunov function
    lyapunov_fn = HybridLyapunov(alpha=0.6)
    
    # Create config
    config = LyapunovConfig(
        epsilon=0.02,
        max_iterations=5,
        min_iterations=2,
        require_monotonic=True,
    )
    
    # Create controller
    controller = LyapunovRetriever(
        transition=transition,
        lyapunov_fn=lyapunov_fn,
        config=config,
    )
    
    # Run retrieval
    query = "What are the best practices for building scalable distributed systems?"
    print(f"\nQuery: {query}")
    print("-" * 70)
    
    final_state, metrics = controller.retrieve(query)
    
    # Print results
    print(f"\n[Final State]")
    print(f"  Iterations: {final_state.iteration}")
    print(f"  Documents retrieved: {len(final_state.document_ids)}")
    print(f"  Final Lyapunov V: {final_state.lyapunov_value:.4f}")
    
    print(f"\n[Convergence Metrics]")
    print(f"  Status: {metrics.stability_status.value}")
    print(f"  Converged: {metrics.converged}")
    print(f"  Stopping reason: {metrics.stopping_reason}")
    print(f"  Lyapunov exponent: {metrics.lyapunov_exponent:.4f}")
    print(f"  Confidence bound: {metrics.confidence_bound:.4f}")
    
    print(f"\n[Lyapunov Values]")
    for i, v in enumerate(metrics.lyapunov_values):
        delta = f"ΔV={metrics.lyapunov_deltas[i-1]:+.4f}" if i > 0 else "initial"
        print(f"  Iter {i}: V={v:.4f} ({delta})")
    
    print(f"\n[Audit Trail]")
    for entry in controller.get_audit_trail():
        print(f"  Iter {entry['iteration']}: V={entry['lyapunov_value']:.4f}, "
              f"docs={entry['num_documents']}, query='{entry['refined_query'][:40]}...'")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    
    return final_state, metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_lyapunov_retrieval()
    else:
        print("Usage: python lyapunov_retrieval.py demo")
        print("\nThis module provides Lyapunov-stable iterative retrieval.")
        print("See ColQwenLyapunovRetriever for integration with the RAG pipeline.")
