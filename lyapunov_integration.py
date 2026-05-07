"""
Lyapunov-Stable Agent Integration
==================================

Connects the Lyapunov-stable iterative retrieval with:
1. The Dr. Zero search agent (Restate workflows)
2. The HRPO training loop
3. Enterprise explainability requirements

Key insight: The Lyapunov framework provides formal guarantees that
connect to the HRPO training objective:

    Training:   Minimize V(x*) by learning better f(x_t, q)
    Inference:  Apply f until V converges

This creates a virtuous cycle:
- Better f (from training) → Faster convergence (lower Lyapunov exponent)
- Convergence data → Better training signal (V as reward proxy)
"""

import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import numpy as np

from lyapunov_retrieval import (
    LyapunovConfig,
    LyapunovRetriever,
    RetrievalTransition,
    RetrievalState,
    ConvergenceMetrics,
    HybridLyapunov,
    RelevanceLyapunov,
    StabilityStatus,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EnterpriseRetrievalConfig:
    """Configuration for enterprise-grade retrieval with stability guarantees"""
    # Lyapunov settings
    lyapunov: LyapunovConfig = None
    
    # Quality requirements
    min_confidence: float = 0.7        # Minimum 1 - V for acceptance
    require_convergence: bool = True   # Must converge to return results
    max_iterations: int = 3            # Lower for latency-sensitive use
    
    # Audit requirements
    full_audit_trail: bool = True
    include_reasoning: bool = True     # Explain why retrieval stopped
    
    def __post_init__(self):
        if self.lyapunov is None:
            self.lyapunov = LyapunovConfig(
                epsilon=0.02,
                max_iterations=self.max_iterations,
                min_iterations=1,
                require_monotonic=True,
                fallback_on_instability=True,
                audit_all_states=True,
            )


# =============================================================================
# HRPO + Lyapunov Training Integration
# =============================================================================

class LyapunovHRPOTrainer:
    """
    Integrates Lyapunov stability analysis with HRPO training.
    
    Key idea: Use Lyapunov exponent as an additional training signal.
    
    - Fast convergence (low λ) → Good retrieval policy
    - Slow/no convergence (high λ) → Needs improvement
    - Instability → Negative training signal
    """
    
    def __init__(self, config: EnterpriseRetrievalConfig):
        self.config = config
        
    def compute_lyapunov_reward(
        self,
        metrics: ConvergenceMetrics,
        user_rating: int,
    ) -> Dict:
        """
        Compute combined reward from Lyapunov metrics and user rating.
        
        R = α * R_user + β * R_convergence + γ * R_stability
        """
        # User rating component (normalized to [0, 1])
        r_user = (user_rating - 1) / 4.0
        
        # Convergence component (based on final V and convergence speed)
        if metrics.converged:
            # Reward for fast convergence
            speed_bonus = 1.0 - (metrics.convergence_iteration or 5) / 5.0
            r_convergence = 0.8 + 0.2 * speed_bonus
        else:
            r_convergence = 0.3  # Partial credit for stability
        
        # Stability component (based on Lyapunov exponent)
        # Negative exponent = converging, positive = diverging
        if metrics.lyapunov_exponent < -0.1:
            r_stability = 1.0  # Strong convergence
        elif metrics.lyapunov_exponent < 0:
            r_stability = 0.7  # Weak convergence
        elif metrics.stability_status == StabilityStatus.STABLE:
            r_stability = 0.5  # Stable but not converging
        else:
            r_stability = 0.0  # Unstable
        
        # Weighted combination
        alpha, beta, gamma = 0.5, 0.3, 0.2
        combined = alpha * r_user + beta * r_convergence + gamma * r_stability
        
        return {
            "r_user": r_user,
            "r_convergence": r_convergence,
            "r_stability": r_stability,
            "combined_reward": combined,
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
        }
    
    def create_training_example(
        self,
        query: str,
        final_state: RetrievalState,
        metrics: ConvergenceMetrics,
        response: str,
        user_rating: int,
    ) -> Dict:
        """
        Create a training example augmented with Lyapunov metrics.
        
        This extends the standard HRPO example with:
        - Convergence trajectory
        - Stability class
        - Lyapunov-augmented reward
        """
        rewards = self.compute_lyapunov_reward(metrics, user_rating)
        
        return {
            "query": query,
            "response": response,
            "context_chunks": final_state.document_contents,
            
            # Standard HRPO fields
            "user_rating": user_rating,
            "hop_count": final_state.iteration + 1,  # Iterations as hop proxy
            
            # Lyapunov-augmented fields
            "lyapunov_trajectory": metrics.lyapunov_values,
            "lyapunov_exponent": metrics.lyapunov_exponent,
            "stability_status": metrics.stability_status.value,
            "converged": metrics.converged,
            "convergence_iteration": metrics.convergence_iteration,
            
            # Combined reward
            "lyapunov_reward": rewards["combined_reward"],
            "reward_components": rewards,
            
            # Metadata
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def group_by_stability(
        self,
        examples: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Group training examples by stability status.
        
        Alternative to hop-based grouping that uses Lyapunov properties.
        """
        groups = {
            "fast_converging": [],    # λ < -0.2
            "slow_converging": [],    # -0.2 ≤ λ < 0
            "marginal": [],           # λ ≈ 0
            "unstable": [],           # λ > 0 or status == unstable
        }
        
        for ex in examples:
            λ = ex.get("lyapunov_exponent", 0)
            status = ex.get("stability_status", "unknown")
            
            if status == "unstable" or λ > 0.1:
                groups["unstable"].append(ex)
            elif λ < -0.2:
                groups["fast_converging"].append(ex)
            elif λ < 0:
                groups["slow_converging"].append(ex)
            else:
                groups["marginal"].append(ex)
        
        return groups
    
    def compute_group_baselines(
        self,
        groups: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """
        Compute baselines for each stability group.
        
        This is the HRPO baseline computation adapted for Lyapunov grouping.
        """
        baselines = {}
        
        for group_name, examples in groups.items():
            if not examples:
                baselines[group_name] = 0.5  # Default
                continue
            
            # Use Lyapunov-augmented reward as the metric
            rewards = [ex.get("lyapunov_reward", 0.5) for ex in examples]
            baselines[group_name] = np.mean(rewards)
        
        return baselines


# =============================================================================
# Enterprise Retrieval Service
# =============================================================================

class EnterpriseRetriever:
    """
    Enterprise-grade retriever with Lyapunov stability guarantees.
    
    Provides:
    1. Formal convergence guarantees
    2. Explainable stopping criteria
    3. Audit trail for compliance
    4. Integration with training loop
    """
    
    def __init__(
        self,
        rag_pipeline,
        query_expander,
        config: Optional[EnterpriseRetrievalConfig] = None,
    ):
        self.rag = rag_pipeline
        self.expander = query_expander
        self.config = config or EnterpriseRetrievalConfig()
        
        # Initialize Lyapunov components
        self.lyapunov_fn = HybridLyapunov(alpha=0.6)
        
        self.transition = RetrievalTransition(
            embedder=self._embed,
            retriever=self._retrieve,
            query_refiner=self._refine_query,
            top_k=getattr(self.rag.config, 'top_k', 5) if hasattr(self.rag, 'config') else 5,
        )
        
        self.controller = LyapunovRetriever(
            transition=self.transition,
            lyapunov_fn=self.lyapunov_fn,
            config=self.config.lyapunov,
        )
        
        # Training integration
        self.trainer = LyapunovHRPOTrainer(self.config)
        
        # Metrics accumulator
        self.search_history: List[Dict] = []
    
    def _embed(self, text: str) -> np.ndarray:
        return np.array(self.rag.embedder.embed(text))
    
    def _retrieve(self, query: str, top_k: int) -> List[Dict]:
        return self.rag.retrieve(query, [])[:top_k]
    
    def _refine_query(self, query: str, docs: List[str]) -> str:
        expanded = self.expander.expand(query)
        if expanded.get("expansions"):
            return expanded["expansions"][0]
        return query
    
    def search(
        self,
        query: str,
        require_convergence: Optional[bool] = None,
    ) -> Dict:
        """
        Perform enterprise-grade search with Lyapunov guarantees.
        """
        require_conv = require_convergence if require_convergence is not None else self.config.require_convergence
        
        # Run retrieval
        initial = self._retrieve(query, self.transition.top_k)
        final_state, metrics = self.controller.retrieve(query, initial)
        
        # Check quality requirements
        confidence = 1.0 - final_state.lyapunov_value
        meets_requirements = (
            confidence >= self.config.min_confidence and
            (not require_conv or metrics.converged)
        )
        
        # Build result
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
            
            # Convergence info
            "converged": metrics.converged,
            "iterations": final_state.iteration,
            "confidence": confidence,
            
            # Quality flags
            "meets_requirements": meets_requirements,
            "stopping_reason": metrics.stopping_reason,
            "stability_status": metrics.stability_status.value,
            
            # Explainability
            "reasoning": self._generate_reasoning(metrics, final_state),
        }
        
        if self.config.full_audit_trail:
            result["audit_trail"] = self.controller.get_audit_trail()
            result["lyapunov_trajectory"] = metrics.lyapunov_values
        
        # Store for training
        self.search_history.append({
            "query": query,
            "result": result,
            "metrics": metrics.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return result
    
    def _generate_reasoning(
        self,
        metrics: ConvergenceMetrics,
        state: RetrievalState
    ) -> str:
        """Generate human-readable explanation of retrieval process"""
        parts = []
        
        # Convergence explanation
        if metrics.converged:
            parts.append(
                f"Search converged after {metrics.convergence_iteration} iterations "
                f"with confidence {1 - metrics.lyapunov_values[-1]:.1%}."
            )
        else:
            parts.append(
                f"Search stopped after {state.iteration} iterations "
                f"({metrics.stopping_reason})."
            )
        
        # Stability explanation
        if metrics.stability_status == StabilityStatus.STABLE:
            parts.append("Results are stable and reliable.")
        elif metrics.stability_status == StabilityStatus.CONVERGING:
            parts.append("Results are improving with each iteration.")
        elif metrics.stability_status == StabilityStatus.MARGINAL:
            parts.append("Results are marginally stable; consider reviewing.")
        else:
            parts.append("⚠️ Results show instability; interpret with caution.")
        
        # Confidence explanation
        confidence = 1 - metrics.lyapunov_values[-1]
        if confidence >= 0.8:
            parts.append(f"High confidence ({confidence:.1%}) in relevance.")
        elif confidence >= 0.6:
            parts.append(f"Moderate confidence ({confidence:.1%}); additional sources may help.")
        else:
            parts.append(f"Low confidence ({confidence:.1%}); results may be incomplete.")
        
        return " ".join(parts)
    
    def record_feedback(
        self,
        query_id: str,
        user_rating: int,
        response: str,
    ) -> Dict:
        """
        Record user feedback and create training example.
        """
        # Find the corresponding search
        search_record = None
        for record in reversed(self.search_history):
            if record["query"] in query_id or hash(record["query"]) % 10000 == int(query_id.split("_")[-1]) % 10000:
                search_record = record
                break
        
        if not search_record:
            return {"error": "Search record not found"}
        
        # Reconstruct state and metrics (simplified - just need key fields)
        result = search_record["result"]
        metrics_dict = search_record["metrics"]
        
        # Create metrics with required fields
        metrics = ConvergenceMetrics(
            lyapunov_values=metrics_dict.get("lyapunov_values", []),
            lyapunov_deltas=metrics_dict.get("lyapunov_deltas", []),
            state_distances=metrics_dict.get("state_distances", []),
            stability_status=StabilityStatus(metrics_dict.get("stability_status", "stable")),
            converged=metrics_dict.get("converged", False),
            convergence_iteration=metrics_dict.get("convergence_iteration"),
            lyapunov_exponent=metrics_dict.get("lyapunov_exponent", 0.0),
            stopping_reason=metrics_dict.get("stopping_reason", ""),
            confidence_bound=metrics_dict.get("confidence_bound", 1.0),
        )
        
        final_state = RetrievalState(
            iteration=result["iterations"],
            document_ids=[d["id"] for d in result["documents"]],
            document_contents=[d["content"] for d in result["documents"]],
            document_scores=[d["score"] for d in result["documents"]],
            original_query=result["query"],
            refined_query=result["query"],
            lyapunov_value=1 - result["confidence"],
        )
        
        # Create training example
        training_example = self.trainer.create_training_example(
            query=result["query"],
            final_state=final_state,
            metrics=metrics,
            response=response,
            user_rating=user_rating,
        )
        
        # Store training example back in history
        search_record["training_example"] = training_example
        
        return {
            "status": "recorded",
            "training_example": training_example,
            "lyapunov_reward": training_example["lyapunov_reward"],
        }
    
    def get_training_batch(self) -> Dict:
        """
        Prepare a training batch from accumulated feedback.
        """
        # Collect all examples with ratings
        examples = [
            record.get("training_example")
            for record in self.search_history
            if record.get("training_example")
        ]
        
        if not examples:
            return {"status": "no_data", "examples": []}
        
        # Group by stability
        groups = self.trainer.group_by_stability(examples)
        baselines = self.trainer.compute_group_baselines(groups)
        
        # Compute relative rewards
        for group_name, group_examples in groups.items():
            baseline = baselines[group_name]
            for ex in group_examples:
                ex["relative_reward"] = ex["lyapunov_reward"] - baseline
        
        return {
            "status": "ready",
            "total_examples": len(examples),
            "groups": {k: len(v) for k, v in groups.items()},
            "baselines": baselines,
            "examples": examples,
        }


# =============================================================================
# Restate Integration
# =============================================================================

def create_lyapunov_agent_handlers():
    """
    Create Restate handlers for Lyapunov-stable retrieval.
    
    Extends the existing search-agent with stability guarantees.
    """
    try:
        import restate
        from restate import Workflow, WorkflowContext
    except ImportError:
        print("Restate not installed")
        return None
    
    lyapunov_agent = Workflow("lyapunov-search-agent")
    
    @lyapunov_agent.main()
    async def stable_search(ctx: WorkflowContext, query_data: dict) -> dict:
        """
        Lyapunov-stable search workflow.
        """
        from rag_pipeline import RAGPipeline, RAGConfig
        from query_expander import QueryExpander
        
        # Initialize components
        rag_config = RAGConfig(use_mock_llm=True)
        rag = RAGPipeline(rag_config)
        expander = QueryExpander(rag_config)
        
        retriever = EnterpriseRetriever(rag, expander)
        
        # Run search
        result = await ctx.run(
            "lyapunov_search",
            lambda: retriever.search(query_data["query"])
        )
        
        # Generate response
        response = await ctx.run(
            "generate_response",
            lambda: rag.generate(query_data["query"], result["documents"])
        )
        
        return {
            "query_id": query_data.get("query_id", ""),
            "response": response["text"],
            "documents": result["documents"],
            "convergence": {
                "converged": result["converged"],
                "iterations": result["iterations"],
                "confidence": result["confidence"],
                "stability": result["stability_status"],
            },
            "reasoning": result["reasoning"],
            "audit_trail": result.get("audit_trail", []),
        }
    
    return lyapunov_agent


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate Lyapunov-stable enterprise retrieval"""
    print("\n" + "=" * 70)
    print("LYAPUNOV-STABLE ENTERPRISE RETRIEVAL DEMO")
    print("=" * 70)
    
    # Import RAG components (use mock versions)
    from rag_pipeline import RAGPipeline, RAGConfig, MockEmbedder
    from query_expander import QueryExpander
    
    # Initialize
    config = RAGConfig(use_mock_llm=True, use_mock_embeddings=True)
    rag = RAGPipeline(config)
    expander = QueryExpander(config)
    
    enterprise_config = EnterpriseRetrievalConfig(
        min_confidence=0.5,  # Lower for demo
        require_convergence=False,  # Don't require for demo
        max_iterations=3,
    )
    
    retriever = EnterpriseRetriever(rag, expander, enterprise_config)
    
    # Example queries
    queries = [
        "What are the best practices for distributed system architecture?",
        "How do transformer attention mechanisms work?",
        "Compare ACID vs BASE consistency models",
    ]
    
    for query in queries:
        print(f"\n{'─' * 70}")
        print(f"Query: {query}")
        print("─" * 70)
        
        result = retriever.search(query)
        
        print(f"\n[Convergence]")
        print(f"  Converged: {result['converged']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Stability: {result['stability_status']}")
        
        print(f"\n[Reasoning]")
        print(f"  {result['reasoning']}")
        
        print(f"\n[Top Documents]")
        for doc in result['documents'][:3]:
            print(f"  - [{doc['score']:.3f}] {doc['content'][:60]}...")
        
        # Simulate feedback
        rating = np.random.randint(3, 6)  # Random 3-5 rating
        feedback_result = retriever.record_feedback(
            query_id=f"q_{hash(query) % 10000}",
            user_rating=rating,
            response="Generated response for query",
        )
        
        print(f"\n[Feedback Recorded]")
        print(f"  User rating: {rating}/5")
        print(f"  Lyapunov reward: {feedback_result.get('lyapunov_reward', 'N/A'):.3f}")
    
    # Get training batch
    print(f"\n{'=' * 70}")
    print("TRAINING BATCH SUMMARY")
    print("=" * 70)
    
    batch = retriever.get_training_batch()
    print(f"\nTotal examples: {batch['total_examples']}")
    print(f"Stability groups: {batch['groups']}")
    print(f"Baselines: {batch['baselines']}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        print("Usage: python lyapunov_integration.py demo")
