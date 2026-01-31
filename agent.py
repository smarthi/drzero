"""
Dr. Zero-inspired Self-Evolving Search Agent with Restate
=========================================================

Complete demo implementation with mock services for local testing.
"""

import restate
from restate import VirtualObject, Workflow, Service, ObjectContext, ObjectSharedContext, WorkflowContext, Context
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime
import json
import hashlib
import time
import os

# Import our components
from rag_pipeline import RAGPipeline, RAGConfig
from query_expander import QueryExpander


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SearchQuery:
    query_id: str
    original_query: str
    user_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class ExpandedQueries:
    original: str
    expansions: list[str]
    sub_queries: list[str]
    hop_count: int

@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)

@dataclass
class SearchResult:
    query_id: str
    original_query: str
    expanded_queries: dict
    retrieved_chunks: list[dict]
    generated_response: str
    model_used: str
    retrieval_latency_ms: float
    generation_latency_ms: float

@dataclass
class UserFeedback:
    query_id: str
    rating: int
    selected_chunks: list[str]
    feedback_text: Optional[str] = None
    user_id: str = "anonymous"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# =============================================================================
# Initialize RAG Pipeline
# =============================================================================

config = RAGConfig(
    opensearch_host=os.getenv("OPENSEARCH_HOST", "localhost"),
    opensearch_port=int(os.getenv("OPENSEARCH_PORT", "9200")),
    use_mock_llm=os.getenv("USE_MOCK_LLM", "true").lower() == "true",
)

rag_pipeline = RAGPipeline(config)
query_expander = QueryExpander(config)


# =============================================================================
# Result Store (in-memory for demo, use Redis/DynamoDB in production)
# =============================================================================

_result_store: dict[str, dict] = {}


def store_result(result: SearchResult) -> dict:
    _result_store[result.query_id] = asdict(result)
    return {"stored": True, "query_id": result.query_id}


def get_stored_result(query_id: str) -> Optional[dict]:
    return _result_store.get(query_id)


# =============================================================================
# Feedback Store - Virtual Object for per-user learning
# =============================================================================

feedback_store = VirtualObject("feedback-store")

@feedback_store.handler()
async def record_feedback(ctx: ObjectContext, feedback_data: dict) -> dict:
    """Record user feedback and update learning signals."""
    feedback = UserFeedback(**feedback_data)
    
    # Get existing feedback history
    history = await ctx.get("feedback_history") or []
    history.append(asdict(feedback))
    
    # Keep last 1000 entries
    if len(history) > 1000:
        history = history[-1000:]
    
    ctx.set("feedback_history", history)

    # Update stats
    stats = await ctx.get("stats") or {
        "total_queries": 0,
        "total_ratings": 0,
        "avg_rating": 0.0,
        "rating_distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
        "helpful_chunk_ids": {},
    }
    
    stats["total_queries"] += 1
    stats["total_ratings"] += feedback.rating
    stats["avg_rating"] = stats["total_ratings"] / stats["total_queries"]
    stats["rating_distribution"][str(feedback.rating)] = \
        stats["rating_distribution"].get(str(feedback.rating), 0) + 1
    
    for chunk_id in feedback.selected_chunks:
        stats["helpful_chunk_ids"][chunk_id] = \
            stats["helpful_chunk_ids"].get(chunk_id, 0) + 1
    
    ctx.set("stats", stats)

    return {"status": "recorded", "new_avg_rating": stats["avg_rating"]}


@feedback_store.handler(kind="shared")
async def get_user_preferences(ctx: ObjectSharedContext) -> dict:
    """Get learned user preferences."""
    stats = await ctx.get("stats") or {}
    history = await ctx.get("feedback_history") or []

    return {
        "avg_rating": stats.get("avg_rating", 0),
        "total_interactions": stats.get("total_queries", 0),
        "rating_distribution": stats.get("rating_distribution", {}),
        "helpful_chunks": stats.get("helpful_chunk_ids", {}),
        "recent_feedback_count": len(history[-100:] if len(history) > 100 else history),
    }


@feedback_store.handler(kind="shared")
async def get_feedback_history(ctx: ObjectSharedContext) -> dict:
    """Get recent feedback history."""
    history = await ctx.get("feedback_history") or []
    return {"history": history[-50:]}  # Return last 50


# =============================================================================
# Global Learning Store - HRPO-inspired grouping
# =============================================================================

learning_store = VirtualObject("learning-store")

@learning_store.handler()
async def record_query_outcome(ctx: ObjectContext, data: dict) -> dict:
    """
    Record query outcomes grouped by hop complexity (HRPO-inspired).
    """
    hop_count = data.get("hop_count", 1)
    rating = data.get("rating", 3)
    query_hash = data.get("query_hash", "unknown")
    query_text = data.get("query_text", "")
    
    # Get hop-grouped outcomes
    hop_groups = await ctx.get("hop_groups") or {}
    
    hop_key = str(hop_count)
    if hop_key not in hop_groups:
        hop_groups[hop_key] = {
            "total_queries": 0,
            "total_rating": 0,
            "avg_rating": 0.0,
            "examples": [],
        }
    
    group = hop_groups[hop_key]
    group["total_queries"] += 1
    group["total_rating"] += rating
    group["avg_rating"] = group["total_rating"] / group["total_queries"]
    
    # Capture timestamp deterministically via ctx.run
    timestamp = await ctx.run("get_timestamp", lambda: datetime.utcnow().isoformat())

    # Keep example queries for analysis
    group["examples"].append({
        "query_hash": query_hash,
        "query_preview": query_text[:100],
        "rating": rating,
        "timestamp": timestamp,
    })

    # Limit examples
    if len(group["examples"]) > 500:
        group["examples"] = group["examples"][-500:]

    ctx.set("hop_groups", hop_groups)
    
    # Compute relative reward
    baseline = group["avg_rating"]
    relative_reward = rating - baseline
    
    return {
        "hop_count": hop_count,
        "absolute_rating": rating,
        "group_baseline": round(baseline, 2),
        "relative_reward": round(relative_reward, 2),
    }


@learning_store.handler(kind="shared")
async def get_learning_signals(ctx: ObjectSharedContext) -> dict:
    """Get aggregated learning signals for evolution."""
    hop_groups = await ctx.get("hop_groups") or {}

    return {
        "hop_groups": hop_groups,
        "ready_for_evolution": any(
            g.get("total_queries", 0) >= 10  # Lower threshold for demo
            for g in hop_groups.values()
        ),
        "total_feedback": sum(g.get("total_queries", 0) for g in hop_groups.values()),
    }


@learning_store.handler(kind="shared")
async def get_evolution_recommendations(ctx: ObjectSharedContext) -> dict:
    """Generate evolution recommendations based on feedback patterns."""
    hop_groups = await ctx.get("hop_groups") or {}
    
    recommendations = []
    
    for hop_count, group in hop_groups.items():
        avg = group.get("avg_rating", 3.0)
        total = group.get("total_queries", 0)
        
        if total < 5:
            continue
            
        if avg < 3.0:
            recommendations.append({
                "hop_count": int(hop_count),
                "issue": "low_satisfaction",
                "avg_rating": round(avg, 2),
                "recommendation": "Increase retrieval candidates and improve query expansion",
            })
        elif avg > 4.0:
            recommendations.append({
                "hop_count": int(hop_count),
                "issue": "optimization_opportunity",
                "avg_rating": round(avg, 2),
                "recommendation": "Can reduce latency by decreasing candidate pool",
            })
    
    return {"recommendations": recommendations}


# =============================================================================
# Search Agent Workflow
# =============================================================================

search_agent = Workflow("search-agent")

@search_agent.main()
async def run_search(ctx: WorkflowContext, query_data: dict) -> dict:
    """
    Main search workflow with durable execution.
    """
    query = SearchQuery(**query_data)
    
    # Step 1: Query Expansion (durable)
    expanded = await ctx.run(
        "expand_query", 
        lambda: query_expander.expand(query.original_query)
    )
    
    # Step 2: RAG Retrieval (durable)
    retrieval_start = time.time()
    chunks = await ctx.run(
        "retrieve_chunks", 
        lambda: rag_pipeline.retrieve(
            query.original_query,
            expanded.get("expansions", [])
        )
    )
    retrieval_latency = (time.time() - retrieval_start) * 1000
    
    # Step 3: LLM Generation (durable)
    gen_start = time.time()
    response = await ctx.run(
        "generate_response", 
        lambda: rag_pipeline.generate(query.original_query, chunks)
    )
    generation_latency = (time.time() - gen_start) * 1000
    
    # Step 4: Store result
    result = SearchResult(
        query_id=query.query_id,
        original_query=query.original_query,
        expanded_queries=expanded,
        retrieved_chunks=chunks,
        generated_response=response["text"],
        model_used=response.get("model", "mock"),
        retrieval_latency_ms=retrieval_latency,
        generation_latency_ms=generation_latency,
    )
    
    await ctx.run("store_result", lambda: store_result(result))
    
    return {
        "query_id": query.query_id,
        "response": response["text"],
        "chunks": chunks,
        "expanded_queries": expanded,
        "latency": {
            "retrieval_ms": round(retrieval_latency, 2),
            "generation_ms": round(generation_latency, 2),
        },
        "status": "complete",
    }


# =============================================================================
# Feedback Service - standalone service to avoid workflow lifecycle issues
# =============================================================================

feedback_service = Service("feedback-service")

@feedback_service.handler()
async def submit_feedback(ctx: Context, feedback_data: dict) -> dict:
    """Handler for user feedback submission."""
    # Build feedback dict deterministically — the timestamp default_factory
    # is non-deterministic, so resolve it via ctx.run to journal the result
    feedback_dict = await ctx.run(
        "build_feedback",
        lambda: asdict(UserFeedback(**feedback_data))
    )

    # Record in user's feedback store — use object_call (not object_send)
    # to wait for the write to complete before returning to the UI
    user_id = feedback_dict.get("user_id") or "anonymous"
    feedback_result = await ctx.object_call(
        record_feedback, key=user_id, arg=feedback_dict
    )

    # Get the original query info
    query_id = feedback_dict["query_id"]
    result_data = await ctx.run(
        "get_stored_result", lambda: get_stored_result(query_id)
    )

    if result_data:
        hop_count = result_data.get("expanded_queries", {}).get("hop_count", 1)
        query_hash = hashlib.md5(
            result_data.get("original_query", "").encode()
        ).hexdigest()[:8]

        # Record in global learning store (HRPO-style)
        learning_result = await ctx.object_call(record_query_outcome, key="global", arg={
            "hop_count": hop_count,
            "rating": feedback_dict["rating"],
            "query_hash": query_hash,
            "query_text": result_data.get("original_query", ""),
        })

        return {
            "status": "feedback_recorded",
            "query_id": query_id,
            "feedback": feedback_result,
            "learning": learning_result,
        }

    return {
        "status": "feedback_recorded",
        "query_id": query_id,
        "feedback": feedback_result,
    }


# =============================================================================
# App Setup
# =============================================================================

app = restate.app(
    services=[search_agent, feedback_store, learning_store, feedback_service]
)


def create_asgi_app():
    """Create ASGI app with health endpoint"""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route, Mount
    
    async def health(request):
        return JSONResponse({"status": "healthy"})
    
    # Mount Restate app and add health endpoint
    routes = [
        Route("/health", health),
        Mount("/", app=app),
    ]
    
    return Starlette(routes=routes)


if __name__ == "__main__":
    import hypercorn.asyncio
    import hypercorn.config
    
    asgi_app = create_asgi_app()
    
    hconfig = hypercorn.config.Config()
    hconfig.bind = ["0.0.0.0:9080"]
    
    print("🚀 Starting Dr. Zero Search Agent on port 9080...")
    print("   Register with Restate: restate deployments register http://localhost:9080")
    
    import asyncio
    asyncio.run(hypercorn.asyncio.serve(asgi_app, hconfig))
