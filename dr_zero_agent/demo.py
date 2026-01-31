#!/usr/bin/env python3
"""
Dr. Zero Search Agent - CLI Demo Script
========================================

Quick demo without Docker/UI - run the agent locally and test.

Usage:
    1. Start Restate: docker run --rm -p 8080:8080 -p 9070:9070 docker.restate.dev/restatedev/restate:latest
    2. In another terminal: python agent.py
    3. In another terminal: python demo.py
"""

import httpx
import uuid
import time
import sys

RESTATE_URL = "http://localhost:8080"
AGENT_URL = "http://localhost:9080"

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)

def print_section(text):
    print(f"\n--- {text} ---")

def register_agent():
    """Register agent with Restate"""
    print_section("Registering agent with Restate")
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                "http://localhost:9070/deployments",
                json={"uri": AGENT_URL}
            )
            if response.status_code == 200 or response.status_code == 201:
                print("✅ Agent registered successfully")
                return True
            elif response.status_code == 409:
                print("✅ Agent already registered")
                return True
            else:
                print(f"⚠️ Registration response: {response.status_code}")
                return True  # May already be registered
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return False

def run_search(query: str, user_id: str = "demo_user") -> dict:
    """Execute a search and return results"""
    query_id = f"q_{uuid.uuid4().hex[:8]}"
    
    print(f"\n📝 Query: {query}")
    print(f"   ID: {query_id}")
    
    start = time.time()
    
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{RESTATE_URL}/search-agent/{query_id}/run_search",
            json={
                "query_id": query_id,
                "original_query": query,
                "user_id": user_id,
            }
        )
        
        elapsed = time.time() - start
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        result["query_id"] = query_id
        result["elapsed"] = elapsed
        return result

def submit_feedback(query_id: str, rating: int, user_id: str = "demo_user"):
    """Submit feedback for a search"""
    
    with httpx.Client(timeout=10.0) as client:
        response = client.post(
            f"{RESTATE_URL}/search-agent/{query_id}/submit_feedback",
            json={
                "query_id": query_id,
                "rating": rating,
                "selected_chunks": [],
                "user_id": user_id,
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Feedback error: {response.text}")
            return None

def get_learning_signals():
    """Get current learning signals"""
    
    with httpx.Client(timeout=5.0) as client:
        response = client.post(
            f"{RESTATE_URL}/learning-store/global/get_learning_signals",
            json={}
        )
        
        if response.status_code == 200:
            return response.json()
        return None

def interactive_demo():
    """Run interactive demo"""
    
    print_header("Dr. Zero Search Agent - Interactive Demo")
    
    # Register
    if not register_agent():
        print("\n⚠️ Could not register agent. Make sure Restate and agent are running:")
        print("   1. docker run --rm -p 8080:8080 -p 9070:9070 docker.restate.dev/restatedev/restate:latest")
        print("   2. python agent.py")
        return
    
    user_id = f"user_{uuid.uuid4().hex[:6]}"
    print(f"\n👤 Session user: {user_id}")
    
    # Sample queries
    queries = [
        "What is ColBERT?",
        "How does late interaction improve retrieval compared to dense embeddings?",
        "Compare RAG and traditional search approaches",
        "What is HRPO in Dr. Zero?",
    ]
    
    print_section("Running sample queries")
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}]")
        
        result = run_search(query, user_id)
        
        if result:
            # Show results
            expanded = result.get("expanded_queries", {})
            print(f"\n   🔄 Hop count: {expanded.get('hop_count', 1)}")
            print(f"   ⏱️  Latency: {result.get('elapsed', 0)*1000:.0f}ms")
            print(f"\n   📄 Response preview:")
            response_text = result.get("response", "")[:300]
            print(f"   {response_text}...")
            
            # Show sources
            chunks = result.get("chunks", [])
            print(f"\n   📚 Sources ({len(chunks)}):")
            for chunk in chunks[:2]:
                print(f"      - {chunk.get('source', 'Unknown')} (score: {chunk.get('score', 0):.3f})")
            
            # Submit feedback (random 3-5 for demo)
            import random
            rating = random.randint(3, 5)
            feedback = submit_feedback(result["query_id"], rating, user_id)
            
            if feedback:
                learning = feedback.get("learning", {})
                print(f"\n   ⭐ Feedback: {rating}/5")
                print(f"   📊 Group baseline: {learning.get('group_baseline', 'N/A')}")
                print(f"   📈 Relative reward: {learning.get('relative_reward', 'N/A')}")
        
        time.sleep(0.5)  # Small delay between queries
    
    # Show final learning stats
    print_section("Learning Statistics (HRPO)")
    
    signals = get_learning_signals()
    if signals:
        print(f"\n   Total feedback: {signals.get('total_feedback', 0)}")
        print(f"   Ready for evolution: {signals.get('ready_for_evolution', False)}")
        
        for hop, data in signals.get("hop_groups", {}).items():
            print(f"\n   Hop-{hop}:")
            print(f"      Queries: {data.get('total_queries', 0)}")
            print(f"      Avg rating: {data.get('avg_rating', 0):.2f}")
    
    print_header("Demo Complete!")
    print("\n💡 Try the interactive UI:")
    print("   docker-compose up")
    print("   Then visit: http://localhost:8501")

def single_query(query: str):
    """Run a single query"""
    print_header("Single Query Mode")
    
    if not register_agent():
        return
    
    result = run_search(query)
    
    if result:
        print(f"\n📄 Response:\n")
        print(result.get("response", "No response"))
        
        print(f"\n📚 Sources:")
        for chunk in result.get("chunks", []):
            print(f"   - {chunk.get('source', 'Unknown')}")
            print(f"     {chunk.get('content', '')[:100]}...")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        single_query(query)
    else:
        # Interactive demo
        interactive_demo()
