"""
Dr. Zero Search Agent - Interactive Demo UI
============================================

Run with: streamlit run ui.py
"""

import streamlit as st
import httpx
import uuid
import json
import os
from datetime import datetime

# Configuration - read from environment or use default for local dev
DEFAULT_RESTATE_URL = os.getenv("RESTATE_URL", "http://localhost:8080")
RESTATE_URL = st.sidebar.text_input("Restate URL", DEFAULT_RESTATE_URL)

st.set_page_config(
    page_title="Dr. Zero Search Agent",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chunk-card {
        background: #f0f2f6;
        color: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .chunk-card p, .chunk-card strong, .chunk-card span {
        color: #1a1a2e;
    }
    .feedback-section {
        background: #e8f5e9;
        color: #1a1a2e;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .metrics-card {
        background: #e3f2fd;
        color: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metrics-card h4, .metrics-card span {
        color: #1a1a2e;
    }
    .hop-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .hop-1 { background: #c8e6c9; color: #2e7d32; }
    .hop-2 { background: #fff9c4; color: #f57f17; }
    .hop-3 { background: #ffcdd2; color: #c62828; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "query_id" not in st.session_state:
    st.session_state.query_id = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"

# Header
st.title("🔍 Dr. Zero Search Agent")
st.markdown("""
A self-evolving RAG system with human-in-the-loop feedback, inspired by 
[Meta's Dr. Zero paper](https://arxiv.org/abs/2601.07055) and built on 
[Restate](https://restate.dev/) for durable execution.
""")

# Sidebar - User info and stats
with st.sidebar:
    st.header("👤 Session Info")
    st.text(f"User ID: {st.session_state.user_id}")
    
    st.divider()
    
    st.header("📊 Learning Stats")
    
    # Fetch learning signals
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.post(
                f"{RESTATE_URL}/learning-store/global/get_learning_signals",
            )
            if response.status_code == 200:
                signals = response.json()
                
                st.metric("Total Feedback", signals.get("total_feedback", 0))
                
                hop_groups = signals.get("hop_groups", {})
                for hop, data in sorted(hop_groups.items()):
                    avg = data.get("avg_rating", 0)
                    count = data.get("total_queries", 0)
                    st.markdown(f"""
                    **Hop-{hop} Queries**  
                    Count: {count} | Avg: {avg:.2f}⭐
                    """)
                
                if signals.get("ready_for_evolution"):
                    st.success("✅ Ready for evolution!")
    except Exception as e:
        st.warning(f"Stats unavailable: {e}")
    
    st.divider()
    
    st.header("ℹ️ About")
    st.markdown("""
    **Key Concepts:**
    - **Query Expansion**: Generates semantic variations
    - **HRPO Grouping**: Groups by complexity (hops)
    - **Feedback Loop**: Your ratings improve results
    """)


# Main content - Tabs
tab_search, tab_history, tab_evolution = st.tabs(["🔍 Search", "📜 History", "🧬 Evolution"])

with tab_search:
    # Search input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., How does late interaction improve retrieval compared to dense embeddings?",
        height=80
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_clicked and query:
        st.session_state.feedback_submitted = False
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        st.session_state.query_id = query_id
        
        with st.spinner("Searching..."):
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        f"{RESTATE_URL}/search-agent/{query_id}/run_search",
                        json={
                            "query_id": query_id,
                            "original_query": query,
                            "user_id": st.session_state.user_id,
                        }
                    )
                    
                    if response.status_code == 200:
                        st.session_state.search_results = response.json()
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.info("Make sure Restate server and agent are running.")
    
    # Display results
    if st.session_state.search_results:
        results = st.session_state.search_results
        
        st.divider()
        
        # Query analysis
        expanded = results.get("expanded_queries", {})
        hop_count = expanded.get("hop_count", 1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            hop_class = f"hop-{min(hop_count, 3)}"
            st.markdown(f"""
            <div class="metrics-card">
                <h4>Complexity</h4>
                <span class="hop-badge {hop_class}">Hop-{hop_count}</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            latency = results.get("latency", {})
            total_ms = latency.get("retrieval_ms", 0) + latency.get("generation_ms", 0)
            st.markdown(f"""
            <div class="metrics-card">
                <h4>Latency</h4>
                <span>{total_ms:.0f}ms</span>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            chunks = results.get("chunks", [])
            st.markdown(f"""
            <div class="metrics-card">
                <h4>Sources</h4>
                <span>{len(chunks)} chunks</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Show expansions
        with st.expander("🔄 Query Expansions", expanded=False):
            st.write("**Original:**", expanded.get("original", query))
            st.write("**Expansions:**")
            for exp in expanded.get("expansions", []):
                st.markdown(f"- {exp}")
            if expanded.get("sub_queries"):
                st.write("**Sub-queries:**")
                for sq in expanded.get("sub_queries", []):
                    st.markdown(f"- {sq}")
        
        st.divider()
        
        # Generated response
        st.subheader("💬 Response")
        st.markdown(results.get("response", "No response generated."))
        
        # Retrieved chunks
        st.subheader("📄 Retrieved Sources")
        
        chunks = results.get("chunks", [])
        selected_chunks = []
        
        for i, chunk in enumerate(chunks):
            with st.container():
                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    is_helpful = st.checkbox("Helpful", key=f"chunk_{i}", label_visibility="collapsed")
                    if is_helpful:
                        selected_chunks.append(chunk["chunk_id"])
                with col2:
                    st.markdown(f"""
                    <div class="chunk-card">
                        <strong>{chunk.get('source', 'Unknown')}</strong>
                        <span style="float:right; color:#666;">Score: {chunk.get('score', 0):.3f}</span>
                        <p style="margin-top:10px;">{chunk.get('content', '')[:500]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Feedback section
        if not st.session_state.feedback_submitted:
            st.divider()
            st.subheader("⭐ Rate this response")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                rating = st.slider(
                    "Rating",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="1=Poor, 5=Excellent"
                )
            with col2:
                feedback_text = st.text_input(
                    "Optional feedback",
                    placeholder="What could be improved?"
                )
            
            if st.button("📤 Submit Feedback", type="primary"):
                with st.spinner("Submitting feedback..."):
                    try:
                        with httpx.Client(timeout=10.0) as client:
                            response = client.post(
                                f"{RESTATE_URL}/feedback-service/submit_feedback",
                                json={
                                    "query_id": st.session_state.query_id,
                                    "rating": rating,
                                    "selected_chunks": selected_chunks,
                                    "feedback_text": feedback_text or None,
                                    "user_id": st.session_state.user_id,
                                }
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.session_state.feedback_submitted = True
                                st.success(f"✅ Feedback recorded! Relative reward: {result.get('learning', {}).get('relative_reward', 'N/A')}")
                                st.rerun()
                            else:
                                st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error submitting feedback: {e}")
        else:
            st.success("✅ Thank you for your feedback!")

with tab_history:
    st.subheader("📜 Your Feedback History")
    
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.post(
                f"{RESTATE_URL}/feedback-store/{st.session_state.user_id}/get_feedback_history",
            )
            
            if response.status_code == 200:
                history = response.json().get("history", [])
                
                if not history:
                    st.info("No feedback history yet. Try some searches!")
                else:
                    for item in reversed(history[-20:]):  # Show last 20
                        with st.container():
                            col1, col2 = st.columns([0.1, 0.9])
                            with col1:
                                st.markdown(f"### {'⭐' * item.get('rating', 0)}")
                            with col2:
                                st.markdown(f"**Query ID:** {item.get('query_id', 'N/A')}")
                                if item.get("feedback_text"):
                                    st.markdown(f"*{item['feedback_text']}*")
                                st.caption(item.get("timestamp", ""))
                            st.divider()
            
            # User stats
            response = client.post(
                f"{RESTATE_URL}/feedback-store/{st.session_state.user_id}/get_user_preferences",
            )
            
            if response.status_code == 200:
                prefs = response.json()
                
                st.subheader("📊 Your Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Searches", prefs.get("total_interactions", 0))
                with col2:
                    st.metric("Avg Rating Given", f"{prefs.get('avg_rating', 0):.2f}")
                with col3:
                    helpful = len(prefs.get("helpful_chunks", {}))
                    st.metric("Helpful Chunks Marked", helpful)
                
                # Rating distribution
                dist = prefs.get("rating_distribution", {})
                if any(dist.values()):
                    st.subheader("Rating Distribution")
                    chart_data = {f"{k}⭐": v for k, v in sorted(dist.items())}
                    st.bar_chart(chart_data)
    except Exception as e:
        st.warning(f"Could not load history: {e}")

with tab_evolution:
    st.subheader("🧬 System Evolution")
    st.markdown("""
    The system continuously learns from feedback using **HRPO** (Hop-Grouped Relative Policy Optimization):
    
    1. **Queries are grouped by complexity** (hop count)
    2. **Group baselines** are computed from average ratings
    3. **Relative rewards** drive improvement decisions
    """)
    
    try:
        with httpx.Client(timeout=5.0) as client:
            # Get learning signals
            response = client.post(
                f"{RESTATE_URL}/learning-store/global/get_learning_signals",
            )
            
            if response.status_code == 200:
                signals = response.json()
                hop_groups = signals.get("hop_groups", {})
                
                st.subheader("📈 HRPO Group Statistics")
                
                for hop, data in sorted(hop_groups.items()):
                    with st.expander(f"Hop-{hop} Queries ({data.get('total_queries', 0)} total)", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Queries", data.get("total_queries", 0))
                            st.metric("Average Rating", f"{data.get('avg_rating', 0):.2f}")
                        with col2:
                            st.markdown("**Recent Examples:**")
                            for ex in data.get("examples", [])[-5:]:
                                st.caption(f"• {ex.get('query_preview', 'N/A')[:50]}... (⭐{ex.get('rating', '?')})")
            
            # Get recommendations
            response = client.post(
                f"{RESTATE_URL}/learning-store/global/get_evolution_recommendations",
            )
            
            if response.status_code == 200:
                recs = response.json().get("recommendations", [])
                
                if recs:
                    st.subheader("💡 Evolution Recommendations")
                    for rec in recs:
                        if rec.get("issue") == "low_satisfaction":
                            st.warning(f"""
                            **Hop-{rec['hop_count']}**: Low satisfaction (avg {rec['avg_rating']})  
                            → {rec['recommendation']}
                            """)
                        elif rec.get("issue") == "optimization_opportunity":
                            st.success(f"""
                            **Hop-{rec['hop_count']}**: High satisfaction (avg {rec['avg_rating']})  
                            → {rec['recommendation']}
                            """)
                else:
                    st.info("No recommendations yet. Keep providing feedback!")
    except Exception as e:
        st.warning(f"Could not load evolution data: {e}")


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    Dr. Zero Search Agent Demo | Built with Restate + Streamlit
</div>
""", unsafe_allow_html=True)
