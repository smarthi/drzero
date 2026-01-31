"""
Evolution Script - Connects Training Loop with the Agent
=========================================================

This script:
1. Fetches accumulated feedback from Restate
2. Retrieves associated query results
3. Prepares HRPO-grouped training data
4. Triggers fine-tuning when enough data is available
5. Updates the agent to use the new model

Usage:
    python evolution.py check     # Check if ready for evolution
    python evolution.py train     # Run SFT training
    python evolution.py train-dpo # Run DPO training
    python evolution.py demo      # Run demo with synthetic data
"""

import asyncio
import json
import httpx
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import training components
from training_loop import (
    TrainingConfig,
    HRPODataPreparer,
    TrainingOrchestrator,
    LoRATrainer,
    TrainingExample,
)


# =============================================================================
# Configuration
# =============================================================================

RESTATE_URL = "http://localhost:8080"
TRAINING_THRESHOLD = 10  # Minimum examples needed (lower for demo)


# =============================================================================
# Data Fetching from Restate
# =============================================================================

async def fetch_learning_signals() -> Dict:
    """Fetch HRPO-grouped learning signals from the learning store"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{RESTATE_URL}/learning-store/global/get_learning_signals",
            json={},
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching learning signals: {response.status_code}")
            return {}


async def fetch_feedback_history(user_id: str = "default") -> List[Dict]:
    """Fetch feedback history from a user's feedback store"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{RESTATE_URL}/feedback-store/{user_id}/get_feedback_history",
            json={},
        )
        
        if response.status_code == 200:
            return response.json().get("history", [])
        else:
            print(f"Error fetching feedback: {response.status_code}")
            return []


async def fetch_all_feedback() -> List[Dict]:
    """
    Fetch feedback from all users.
    
    In a production system, you'd have a dedicated aggregation store.
    For this demo, we try common user IDs.
    """
    all_feedback = []
    
    # Try common user IDs
    user_ids = ["default", "demo_user", "anonymous", "batch_user"]
    
    for user_id in user_ids:
        feedback = await fetch_feedback_history(user_id)
        all_feedback.extend(feedback)
    
    return all_feedback


# =============================================================================
# Result Store Integration
# =============================================================================

# In-memory cache of query results (in production, use Redis/DynamoDB)
_result_cache: Dict[str, Dict] = {}


def cache_result(query_id: str, result: Dict):
    """Cache a query result"""
    _result_cache[query_id] = result


def get_cached_result(query_id: str) -> Optional[Dict]:
    """Get a cached result"""
    return _result_cache.get(query_id)


async def fetch_query_result(query_id: str) -> Optional[Dict]:
    """
    Fetch a query result.
    
    Note: The current implementation stores results in-memory in agent.py.
    In production, use a persistent store like Redis or DynamoDB.
    """
    # Check cache first
    cached = get_cached_result(query_id)
    if cached:
        return cached
    
    # In production, you'd fetch from your persistent store here
    # For now, return None if not cached
    return None


# =============================================================================
# Evolution Check
# =============================================================================

async def check_evolution_readiness() -> Dict:
    """Check if we have enough data to trigger evolution"""
    signals = await fetch_learning_signals()
    
    if not signals:
        return {
            "ready": False,
            "reason": "Could not fetch learning signals",
        }
    
    hop_groups = signals.get("hop_groups", {})
    total_feedback = signals.get("total_feedback", 0)
    
    print("\n" + "=" * 60)
    print("EVOLUTION READINESS CHECK")
    print("=" * 60)
    
    print(f"\nTotal feedback collected: {total_feedback}")
    print(f"Ready threshold: {TRAINING_THRESHOLD}")
    
    ready_groups = []
    
    print("\nHop Group Analysis:")
    for hop_count, group in hop_groups.items():
        total = group.get("total_queries", 0)
        avg_rating = group.get("avg_rating", 0)
        
        status = "✓ READY" if total >= TRAINING_THRESHOLD else f"Need {TRAINING_THRESHOLD - total} more"
        print(f"  Hop-{hop_count}: {total} queries, avg rating: {avg_rating:.2f} [{status}]")
        
        if total >= TRAINING_THRESHOLD:
            ready_groups.append(int(hop_count))
    
    is_ready = len(ready_groups) > 0
    
    print("\n" + "-" * 60)
    if is_ready:
        print("✓ READY FOR EVOLUTION!")
        print(f"  Ready hop groups: {ready_groups}")
    else:
        print("✗ NOT READY - Need more feedback data")
    print("=" * 60)
    
    return {
        "ready": is_ready,
        "ready_groups": ready_groups,
        "total_feedback": total_feedback,
        "hop_groups": hop_groups,
    }


# =============================================================================
# Training Trigger
# =============================================================================

async def trigger_evolution(
    training_type: str = "sft",
    config: Optional[TrainingConfig] = None
) -> Dict:
    """
    Trigger an evolution cycle.
    
    1. Fetch all feedback
    2. Prepare HRPO-grouped data
    3. Run training (SFT or DPO)
    4. Return results
    """
    config = config or TrainingConfig()
    
    print("\n" + "=" * 60)
    print(f"TRIGGERING {training_type.upper()} EVOLUTION")
    print("=" * 60)
    
    # Step 1: Fetch feedback
    print("\n[1/4] Fetching feedback data...")
    feedback_data = await fetch_all_feedback()
    print(f"      Found {len(feedback_data)} feedback records")
    
    if not feedback_data:
        return {"status": "error", "message": "No feedback data available"}
    
    # Step 2: Build query results map
    # In a real system, you'd fetch these from your persistent store
    print("\n[2/4] Building query results map...")
    
    # For demo, create synthetic results based on feedback
    query_results = {}
    for fb in feedback_data:
        query_id = fb.get("query_id")
        if query_id and query_id not in query_results:
            # Create synthetic result (in production, fetch from store)
            query_results[query_id] = {
                "original_query": f"Query for {query_id}",
                "generated_response": f"Response for query {query_id}. This is a helpful answer.",
                "retrieved_chunks": [
                    {"content": f"Context chunk 1 for {query_id}"},
                    {"content": f"Context chunk 2 for {query_id}"},
                ],
                "expanded_queries": {
                    "hop_count": 1 + (hash(query_id) % 3),  # Synthetic hop count
                    "expansions": [f"Expanded query for {query_id}"],
                },
            }
    
    print(f"      Built results for {len(query_results)} queries")
    
    # Step 3: Prepare HRPO-grouped data
    print("\n[3/4] Preparing HRPO-grouped training data...")
    
    preparer = HRPODataPreparer(config)
    prepared = preparer.prepare_from_feedback(feedback_data, query_results)
    
    print(f"      Positive examples: {len(prepared['positive_examples'])}")
    print(f"      Negative examples: {len(prepared['negative_examples'])}")
    print(f"      Group stats: {prepared['group_stats']}")
    
    # Step 4: Run training
    print(f"\n[4/4] Running {training_type.upper()} training...")
    
    orchestrator = TrainingOrchestrator(config)
    
    if training_type == "dpo":
        result = orchestrator.run_dpo_training(prepared)
    else:
        result = orchestrator.run_sft_training(prepared)
    
    print("\n" + "-" * 60)
    print(f"Evolution Result: {result.get('status', 'unknown')}")
    if result.get('adapter_path'):
        print(f"Adapter saved to: {result['adapter_path']}")
    print("=" * 60)
    
    return result


# =============================================================================
# Demo Mode
# =============================================================================

async def run_demo():
    """Run a complete demo with synthetic data"""
    print("\n" + "=" * 60)
    print("DR. ZERO EVOLUTION DEMO")
    print("=" * 60)
    
    # Create synthetic feedback with variety in ratings
    demo_feedback = []
    
    # Simulate different hop complexities with different rating distributions
    queries = [
        # Simple queries (hop=1) - generally higher ratings
        ("q001", "What is machine learning?", 5, 1),
        ("q002", "Define neural network", 4, 1),
        ("q003", "What is deep learning?", 5, 1),
        ("q004", "Explain AI basics", 4, 1),
        ("q005", "What is NLP?", 3, 1),
        ("q006", "Define transformer", 5, 1),
        ("q007", "What is embedding?", 4, 1),
        ("q008", "Explain attention", 2, 1),  # Low rating - for contrast
        ("q009", "What is BERT?", 5, 1),
        ("q010", "Define RAG", 4, 1),
        
        # Medium queries (hop=2) - mixed ratings
        ("q011", "Compare BERT vs GPT for embeddings", 4, 2),
        ("q012", "How does attention improve retrieval?", 3, 2),
        ("q013", "Difference between dense and sparse retrieval", 5, 2),
        ("q014", "ColBERT vs DPR comparison", 4, 2),
        ("q015", "Late interaction vs cross-encoder", 3, 2),
        ("q016", "Query expansion techniques comparison", 2, 2),  # Low rating
        ("q017", "Hybrid search implementation", 5, 2),
        ("q018", "Reranking strategies comparison", 4, 2),
        ("q019", "BM25 vs semantic search tradeoffs", 3, 2),
        ("q020", "Multi-vector vs single-vector retrieval", 4, 2),
        
        # Complex queries (hop=3) - lower ratings (harder)
        ("q021", "Implement ColBERT with MaxSim for multi-hop QA", 3, 3),
        ("q022", "Compare HRPO vs GRPO for agent training", 2, 3),
        ("q023", "Self-evolution in search agents explained", 4, 3),
        ("q024", "Dr. Zero proposer-solver co-evolution", 3, 3),
        ("q025", "Durable execution for multi-turn search", 4, 3),
        ("q026", "LoRA fine-tuning for retrieval models", 2, 3),  # Low rating
        ("q027", "DPO vs PPO for agent optimization", 3, 3),
        ("q028", "Token-level late interaction mathematics", 1, 3),  # Very low
        ("q029", "RLHF for search ranking improvement", 4, 3),
        ("q030", "Compute-optimal training for search agents", 3, 3),
    ]
    
    query_results = {}
    
    for query_id, query_text, rating, hop_count in queries:
        demo_feedback.append({
            "query_id": query_id,
            "rating": rating,
            "selected_chunks": [],
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        query_results[query_id] = {
            "original_query": query_text,
            "generated_response": f"Here's a comprehensive answer about {query_text.lower()}. "
                                 f"The key points to understand are the underlying concepts "
                                 f"and their practical applications in AI systems.",
            "retrieved_chunks": [
                {"content": f"Relevant context for: {query_text}"},
                {"content": f"Additional information about {query_text.split()[0]}"},
            ],
            "expanded_queries": {
                "hop_count": hop_count,
                "expansions": [f"What is {query_text.split()[-1]}?"],
            },
        }
    
    print(f"\nCreated {len(demo_feedback)} synthetic feedback records")
    print(f"Hop distribution: 1={sum(1 for q in queries if q[3]==1)}, "
          f"2={sum(1 for q in queries if q[3]==2)}, "
          f"3={sum(1 for q in queries if q[3]==3)}")
    
    # Prepare HRPO data
    print("\n[Preparing HRPO-Grouped Data]")
    config = TrainingConfig(min_samples_per_group=5)
    preparer = HRPODataPreparer(config)
    prepared = preparer.prepare_from_feedback(demo_feedback, query_results)
    
    print(f"\nTotal processed: {prepared['total_processed']}")
    print(f"Positive examples (above baseline): {len(prepared['positive_examples'])}")
    print(f"Negative examples (below baseline): {len(prepared['negative_examples'])}")
    print(f"\nGroup Statistics:")
    
    for hop, stats in prepared['group_stats'].items():
        print(f"  Hop-{hop}:")
        print(f"    Total: {stats['total_examples']}")
        print(f"    Baseline: {stats['baseline']:.2f}")
        print(f"    Positive: {stats['positive_count']}")
        print(f"    Negative: {stats['negative_count']}")
    
    print(f"\nReady for SFT: {prepared['ready_for_sft']}")
    print(f"Ready for DPO: {prepared['ready_for_dpo']}")
    
    # Show example positive/negative examples
    if prepared['positive_examples']:
        print("\n[Sample Positive Example]")
        pos_ex = prepared['positive_examples'][0]
        print(f"  Query: {pos_ex.query}")
        print(f"  Rating: {pos_ex.rating}")
        print(f"  Relative Reward: {pos_ex.relative_reward:.2f}")
        print(f"  Hop Count: {pos_ex.hop_count}")
    
    if prepared['negative_examples']:
        print("\n[Sample Negative Example]")
        neg_ex = prepared['negative_examples'][0]
        print(f"  Query: {neg_ex.query}")
        print(f"  Rating: {neg_ex.rating}")
        print(f"  Relative Reward: {neg_ex.relative_reward:.2f}")
        print(f"  Hop Count: {neg_ex.hop_count}")
    
    # Create SFT dataset
    if prepared['ready_for_sft']:
        print("\n[Creating SFT Dataset]")
        sft_data = preparer.create_sft_dataset(prepared['positive_examples'])
        print(f"  Created {len(sft_data)} training examples")
        
        # Save to file
        output_path = Path("./demo_training_data.json")
        with open(output_path, "w") as f:
            json.dump({
                "metadata": {
                    "created": datetime.utcnow().isoformat(),
                    "total_examples": len(sft_data),
                    "group_stats": prepared['group_stats'],
                },
                "sft_data": sft_data,
            }, f, indent=2, default=str)
        print(f"  Saved to: {output_path}")
    
    # Create DPO dataset
    if prepared['ready_for_dpo']:
        print("\n[Creating DPO Dataset]")
        dpo_pairs = preparer.create_dpo_dataset(
            prepared['positive_examples'],
            prepared['negative_examples']
        )
        print(f"  Created {len(dpo_pairs)} preference pairs")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nTo run actual training:")
    print("  1. Install dependencies: pip install torch transformers peft trl")
    print("  2. Run: python training_loop.py --mode sft --data demo_training_data.json")
    
    return prepared


# =============================================================================
# Main CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Dr. Zero Evolution Manager")
    parser.add_argument("command", choices=["check", "train", "train-dpo", "demo"],
                       help="Command to run")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Base model for training")
    parser.add_argument("--output", default="./evolution_output",
                       help="Output directory")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        base_model=args.model,
        output_dir=args.output,
    )
    
    if args.command == "check":
        asyncio.run(check_evolution_readiness())
        
    elif args.command == "train":
        asyncio.run(trigger_evolution("sft", config))
        
    elif args.command == "train-dpo":
        asyncio.run(trigger_evolution("dpo", config))
        
    elif args.command == "demo":
        asyncio.run(run_demo())


if __name__ == "__main__":
    main()
