# Dr. Zero Search Agent 🔍

A self-evolving RAG (Retrieval-Augmented Generation) system with human-in-the-loop feedback, inspired by [Meta's Dr. Zero paper](https://arxiv.org/abs/2601.07055) and built on [Restate](https://restate.dev/) for durable execution.

![Demo](https://img.shields.io/badge/demo-ready-brightgreen) ![Restate](https://img.shields.io/badge/Restate-0.4+-blue) ![Python](https://img.shields.io/badge/Python-3.11+-yellow)

## 🚀 Quick Start (Docker)

```bash
# Clone and start everything
docker-compose up --build

# Wait for "Demo ready!" message, then open:
# 🌐 http://localhost:8501
```

That's it! The UI will be available at http://localhost:8501

## 🎮 What You Can Demo

### 1. Search & Feedback Loop
- Enter queries in the web UI
- See query expansion and retrieved sources
- Rate responses (1-5 stars)
- Watch learning signals accumulate

### 2. HRPO in Action
- Simple queries (1-hop) vs complex (2-3 hops)
- Group baselines computed per complexity
- Relative rewards shown after each feedback

### 3. Durable Execution
- Crash recovery (try stopping/starting agent)
- State persists across restarts
- Each step is checkpointed

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Restate Durable Workflow                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Query → Expand → Retrieve → Generate → [Await Feedback]      │
│                                              │                  │
│                                              ▼                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Feedback Store (per user)                  │  │
│   │   • Rating history, helpful chunks, preferences         │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                              │                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │           Learning Store (global, HRPO)                 │  │
│   │   • Hop-grouped outcomes, baselines, rewards           │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🏃 Running Locally (Without Docker)

### Prerequisites
- Python 3.11+
- Docker (for Restate server)

### Steps

```bash
# Terminal 1: Start Restate
docker run --rm -p 8080:8080 -p 9070:9070 -p 9071:9071 \
    docker.restate.dev/restatedev/restate:latest

# Terminal 2: Install deps and run agent
pip install -r requirements.txt
python agent.py

# Terminal 3: Register and run demo
python demo.py

# Or start the UI
streamlit run ui.py
```

## 📝 API Usage

### Search

```bash
curl -X POST http://localhost:8080/search-agent/my-query-id/run_search \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "my-query-id",
    "original_query": "How does ColBERT work?",
    "user_id": "user123"
  }'
```

### Submit Feedback

```bash
curl -X POST http://localhost:8080/search-agent/my-query-id/submit_feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "my-query-id",
    "rating": 4,
    "selected_chunks": ["chunk_id_1"],
    "user_id": "user123"
  }'
```

### Get Learning Signals

```bash
curl -X POST http://localhost:8080/learning-store/global/get_learning_signals \
  -H "Content-Type: application/json" -d '{}'
```

## 🧠 Key Concepts from Dr. Zero

### 1. Proposer-Solver Co-Evolution
In Dr. Zero, a "proposer" generates questions to train a "solver". In our adaptation:
- **Solver**: The RAG pipeline (query expansion → retrieval → generation)
- **Proposer**: Real users providing diverse queries
- **Reward**: User ratings (1-5 stars)

### 2. HRPO (Hop-Grouped Relative Policy Optimization)
Queries are grouped by complexity (hop count):
- **1-hop**: Simple factual ("What is X?")
- **2-hop**: Synthesis/comparison ("How does X compare to Y?")
- **3+-hop**: Multi-step reasoning ("Explain the relationship between X, Y, and Z")

Each group has its own baseline, and **relative reward** = user_rating - group_baseline

### 3. Durable Execution (Restate)
- Each workflow step is checkpointed
- Crashes trigger automatic recovery
- Human-in-the-loop via durable promises

## 📁 Project Structure

```
dr_zero_agent/
├── agent.py           # Restate workflows & virtual objects
├── rag_pipeline.py    # RAG with mock/real backends
├── query_expander.py  # Query expansion & hop detection
├── ui.py              # Streamlit web interface
├── demo.py            # CLI demo script
├── docker-compose.yml # Full stack deployment
├── Dockerfile.agent   # Agent container
├── Dockerfile.ui      # UI container
└── requirements.txt   # Python dependencies
```

## 🔧 Configuration

Environment variables for the agent:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_LLM` | `true` | Use mock LLM (set `false` for real) |
| `USE_MOCK_EMBEDDINGS` | `true` | Use mock embeddings |
| `OPENSEARCH_HOST` | `localhost` | OpenSearch host |
| `OPENSEARCH_PORT` | `9200` | OpenSearch port |

## 🔌 Integrating Real Services

Replace mock components with your infrastructure:

### ColQwen2 + SageMaker
```python
# In rag_pipeline.py, modify RAGConfig:
config = RAGConfig(
    use_mock_llm=False,
    use_mock_embeddings=False,
    colqwen_endpoint="your-sagemaker-endpoint",
    llm_endpoint="your-llm-endpoint",
)
```

### OpenSearch
```python
config = RAGConfig(
    opensearch_host="your-opensearch.us-east-1.es.amazonaws.com",
    opensearch_port=443,
    opensearch_index="your-documents",
)
```

## 📈 Evolution Recommendations

The system provides recommendations based on feedback patterns:

- **Low satisfaction** (avg < 3): Increase retrieval candidates
- **High satisfaction** (avg > 4): Optimize for latency
- **Per hop-group**: Different strategies for simple vs complex queries

## 🧪 Testing

```bash
# Run the demo
python demo.py

# Single query
python demo.py "What is transformer attention?"

# With pytest
pytest test_agent.py -v
```

## 📚 References

- [Dr. Zero Paper](https://arxiv.org/abs/2601.07055) - Meta's self-evolving search agents
- [Restate Documentation](https://docs.restate.dev/) - Durable execution framework
- [ColBERT](https://arxiv.org/abs/2004.12832) - Late interaction retrieval
- [HRPO](https://arxiv.org/abs/2601.07055) - Hop-grouped relative policy optimization

## 📜 License

MIT

---

Built with ❤️ using [Restate](https://restate.dev) and inspired by [Dr. Zero](https://arxiv.org/abs/2601.07055)
