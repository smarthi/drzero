# Dr. Zero Search Agent 🔍

A self-evolving RAG (Retrieval-Augmented Generation) system with human-in-the-loop feedback, inspired by [Meta's Dr. Zero paper](https://arxiv.org/abs/2601.07055) and built on [Restate](https://restate.dev/) for durable execution.

**New in v2:** Lyapunov-stable iterative retrieval with formal convergence guarantees — critical for clinical and high-stakes applications.

![Demo](https://img.shields.io/badge/demo-ready-brightgreen) ![Restate](https://img.shields.io/badge/Restate-0.4+-blue) ![Python](https://img.shields.io/badge/Python-3.11+-yellow) ![Lyapunov](https://img.shields.io/badge/Lyapunov-stable-purple)

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

### 4. Lyapunov-Stable Retrieval (NEW)
- Formal convergence guarantees for iterative retrieval
- Explainable stopping criteria ("stopped because ΔV < ε")
- Clinical-grade audit trails
- Stability classification (converging, stable, oscillating, unstable)

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Restate Durable Workflow                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Query → Expand → ┌──────────────────────────────────┐ → Generate      │
│                    │  LYAPUNOV ITERATIVE RETRIEVAL    │                 │
│                    │                                  │                 │
│                    │  x_{t+1} = f(x_t, q)            │                 │
│                    │  V(x_{t+1}) ≤ V(x_t)  ✓         │                 │
│                    │  Stop when ΔV < ε               │                 │
│                    └──────────────────────────────────┘                 │
│                                              │                          │
│                                              ▼                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │              Feedback Store (per user)                          │  │
│   │   • Rating history, helpful chunks, preferences                 │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                              │                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │           Learning Store (global, HRPO + Lyapunov)              │  │
│   │   • Hop-grouped outcomes, baselines, rewards                    │  │
│   │   • Lyapunov exponents, stability classification                │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                              │                          │
│                                              ▼                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    Training Pipeline                            │  │
│   │   • LoRA fine-tuning (SFT / DPO)                               │  │
│   │   • Lyapunov-augmented rewards                                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
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

---

## 🔬 Lyapunov-Stable Iterative Retrieval

For clinical and high-stakes applications requiring **formal convergence guarantees**, the system includes a Lyapunov stability framework that frames multi-hop retrieval as a discrete dynamical system.

### Mathematical Framework

```
State:       x_t = (D_t, q_t, r_t)     # documents, refined query, relevance scores
Transition:  x_{t+1} = f(x_t, q₀)      # retrieval-refinement operator  
Lyapunov:    V(x_t) = d(x_t, x*)       # distance to oracle retrieval state
Stability:   V(x_{t+1}) ≤ V(x_t)       # monotonic improvement (Lyapunov condition)
Stopping:    |ΔV| < ε  or  t > T_max   # principled termination
```

### Why This Matters

| Guarantee | Mathematical Basis | Practical Benefit |
|-----------|-------------------|-------------------|
| **Convergence** | V(x_t) monotonically decreases | Retrieval won't oscillate or diverge |
| **Stopping Criterion** | Stop when ΔV < ε | Explainable "why we stopped here" |
| **Confidence Bound** | 1 - V(x_final) | Quantified retrieval quality |
| **Architecture Comparison** | Lyapunov exponent λ | Compare retrieval policies objectively |
| **Audit Trail** | Full state history {x₀, x₁, ...} | Clinical compliance & explainability |

### Lyapunov Functions

The system provides multiple Lyapunov function choices:

```python
# 1. Relevance-based (tracks retrieval quality)
V_relevance(x) = 1 - weighted_avg(document_scores)

# 2. Coverage-based (tracks query facet coverage)
V_coverage(x) = 1 - cosine(query_embedding, doc_set_centroid)

# 3. Hybrid (recommended for clinical use)
V_hybrid(x) = α · V_relevance + (1-α) · V_coverage

# 4. Oracle (for training/evaluation with ground truth)
V_oracle(x) = 1 - Jaccard(retrieved_docs, ground_truth_docs)
```

### Usage

```python
from lyapunov_integration import ClinicalRetriever, ClinicalRetrievalConfig

# Configure for clinical use
config = ClinicalRetrievalConfig(
    min_confidence=0.7,        # Require 70% confidence to accept results
    require_convergence=True,  # Must converge (no early termination)
    max_clinical_iterations=3, # Latency bound for real-time use
)

# Initialize retriever
retriever = ClinicalRetriever(rag_pipeline, query_expander, config)

# Run search with stability guarantees
result = retriever.search("Treatment guidelines for acute myocardial infarction")

# Inspect convergence
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Stability: {result['stability_status']}")
print(f"Reasoning: {result['reasoning']}")

# Full audit trail for compliance
for state in result['audit_trail']:
    print(f"  Iter {state['iteration']}: V={state['lyapunov_value']:.4f}")
```

### Stability Classification

The system classifies retrieval trajectories by their Lyapunov behavior:

| Status | Lyapunov Exponent (λ) | Meaning |
|--------|----------------------|---------|
| `STABLE` | λ ≈ 0, V at minimum | Converged to optimal |
| `CONVERGING` | λ < 0 | Improving each iteration |
| `MARGINAL` | λ ≈ 0, V not at minimum | Stable but not converging |
| `OSCILLATING` | Sign changes in ΔV | Cycling between states |
| `UNSTABLE` | λ > 0 | Diverging (rejected) |

### Demo

```bash
# Run Lyapunov retrieval demo
python lyapunov_retrieval.py demo

# Run clinical integration demo  
python lyapunov_integration.py demo
```

Output:
```
======================================================================
LYAPUNOV-STABLE ITERATIVE RETRIEVAL DEMO
======================================================================

Query: What are the clinical guidelines for treating acute MI?
----------------------------------------------------------------------

[Convergence Metrics]
  Status: stable
  Converged: True
  Stopping reason: converged_at_iter_2_delta_v=-0.0141
  Lyapunov exponent: -0.0360
  Confidence bound: 0.6535

[Lyapunov Values]
  Iter 0: V=0.7281 (initial)
  Iter 1: V=0.6676 (ΔV=-0.0605)
  Iter 2: V=0.6535 (ΔV=-0.0141)  ← Converged!
```

## 📁 Project Structure

```
dr_zero_agent/
├── agent.py               # Restate workflows & virtual objects
├── rag_pipeline.py        # RAG with mock/real backends
├── query_expander.py      # Query expansion & hop detection
├── ui.py                  # Streamlit web interface
├── demo.py                # CLI demo script
│
├── lyapunov_retrieval.py  # 🆕 Lyapunov stability framework
├── lyapunov_integration.py# 🆕 Clinical retriever + training integration
├── training_loop.py       # LoRA fine-tuning (SFT/DPO)
├── evolution.py           # Training orchestration
│
├── docker-compose.yml     # Full stack deployment
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_LLM` | `true` | Use mock LLM (set `false` for real) |
| `USE_MOCK_EMBEDDINGS` | `true` | Use mock embeddings |
| `OPENSEARCH_HOST` | `localhost` | OpenSearch host |
| `OPENSEARCH_PORT` | `9200` | OpenSearch port |

### Lyapunov Configuration

```python
from lyapunov_retrieval import LyapunovConfig

config = LyapunovConfig(
    epsilon=0.02,              # Convergence threshold: |ΔV| < ε
    max_iterations=5,          # Maximum retrieval iterations
    min_iterations=1,          # Minimum before checking convergence
    stability_margin=0.0,      # α: require V_{t+1} ≤ V_t - α||Δx||²
    require_monotonic=True,    # Reject iterations that increase V
    fallback_on_instability=True,  # Revert to last stable state
    audit_all_states=True,     # Keep full history for explainability
)
```

### Clinical Retrieval Configuration

```python
from lyapunov_integration import ClinicalRetrievalConfig

config = ClinicalRetrievalConfig(
    min_confidence=0.7,        # Minimum 1 - V for acceptance
    require_convergence=True,  # Must converge to return results
    max_clinical_iterations=3, # Lower for latency-sensitive clinical use
    full_audit_trail=True,     # Keep full state history
    include_reasoning=True,    # Explain why retrieval stopped
)
```

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

# Lyapunov stability demo
python lyapunov_retrieval.py demo

# Clinical retrieval demo
python lyapunov_integration.py demo

# Training demo (synthetic data)
python evolution.py demo

# With pytest
pytest test_agent.py -v
```

## 🧠 Real Training Loop (LLM Fine-tuning)

The system includes a complete training pipeline that uses accumulated feedback to actually improve the LLM:

### How It Works

1. **HRPO Data Preparation**: Feedback is grouped by query complexity (hop count)
2. **Relative Rewards**: Each response's rating is compared to its group's baseline
3. **Training Selection**: Responses above baseline become positive examples
4. **LoRA Fine-tuning**: Efficient adapter training without modifying base weights

### Training Commands

```bash
# Check if dependencies are installed
python training_loop.py --mode check

# Run demo with synthetic data (no GPU needed)
python evolution.py demo

# Check evolution readiness (needs Restate running)
python evolution.py check

# Trigger SFT training when ready
python evolution.py train

# Trigger DPO training (preference learning)
python evolution.py train-dpo
```

### Install Training Dependencies

```bash
pip install torch transformers peft trl datasets accelerate
```

### Training Architecture

```
                    ┌─────────────────────────────────┐
                    │     User Queries + Ratings      │
                    └───────────────┬─────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                   LYAPUNOV-STABLE RETRIEVAL                       │
│                                                                   │
│   x₀ → f(x₀,q) → x₁ → f(x₁,q) → x₂ → ... → x*                   │
│        │              │              │                            │
│        ▼              ▼              ▼                            │
│      V(x₀)         V(x₁)         V(x₂)    [must decrease!]       │
└───────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                      HRPO + LYAPUNOV TRAINING                     │
│                                                                   │
│   Combined Reward:                                                │
│   R = 0.5 · R_user + 0.3 · R_convergence + 0.2 · R_stability     │
│                                                                   │
│   Grouping Options:                                               │
│   • By hop count (HRPO): 1-hop, 2-hop, 3+-hop                    │
│   • By stability (Lyapunov): fast_converging, slow, marginal     │
│                                                                   │
│   Baseline: μ_group = mean(R | group)                            │
│   Relative Reward: R_rel = R - μ_group                           │
│                                                                   │
│   Training Data:                                                  │
│   • Positive: R_rel > 0.3  →  SFT examples                       │
│   • Negative: R_rel < -0.3 →  DPO rejection examples             │
└───────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                    LoRA FINE-TUNING                               │
│                                                                   │
│   Base Model: Qwen2.5-1.5B-Instruct (configurable)               │
│   Method: LoRA (r=16, α=32) via PEFT                             │
│   Modes: SFT (supervised) or DPO (preference)                    │
└───────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │   Improved f(x,q) Operator      │
                    │   → Lower Lyapunov exponent λ   │
                    │   → Faster convergence          │
                    │   → Better retrieval quality    │
                    └─────────────────────────────────┘
```

### Key Concepts

| Concept | Implementation |
|---------|---------------|
| **HRPO Grouping** | Queries grouped by hop count (1, 2, 3+) |
| **Stability Grouping** | Queries grouped by Lyapunov exponent (fast/slow converging, marginal) |
| **Group Baseline** | Mean reward per group |
| **Relative Reward** | `rating - baseline` |
| **Lyapunov Reward** | `0.5·R_user + 0.3·R_convergence + 0.2·R_stability` |
| **Positive Examples** | `relative_reward > 0.3` (for SFT) |
| **Preference Pairs** | Positive vs negative examples (for DPO) |

### Training Configuration

Edit `training_loop.py` to customize:

```python
@dataclass
class TrainingConfig:
    # Model
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    
    # Training
    learning_rate: float = 2e-4
    batch_size: int = 4
    num_epochs: int = 3
    
    # HRPO thresholds
    positive_reward_threshold: float = 0.3
    negative_reward_threshold: float = -0.3
```

## 📚 References

### Core Papers
- [Dr. Zero](https://arxiv.org/abs/2601.07055) - Meta's self-evolving search agents (2025)
- [HRPO](https://arxiv.org/abs/2601.07055) - Hop-grouped relative policy optimization

### Retrieval
- [ColBERT](https://arxiv.org/abs/2004.12832) - Late interaction retrieval
- [ColBERTv2](https://arxiv.org/abs/2112.01488) - Efficient late interaction

### Training
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
- [DPO](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization

### Infrastructure
- [Restate Documentation](https://docs.restate.dev/) - Durable execution framework

### Stability Theory
- [Lyapunov Stability](https://en.wikipedia.org/wiki/Lyapunov_stability) - Control theory foundations
- [Discrete Lyapunov Functions](https://www.sciencedirect.com/topics/mathematics/lyapunov-function) - For discrete dynamical systems

## 📜 License

MIT

---

Built with ❤️ using [Restate](https://restate.dev) and inspired by [Dr. Zero](https://arxiv.org/abs/2601.07055)

**Key Innovation:** Formal convergence guarantees via Lyapunov stability theory, enabling production-safe iterative retrieval for clinical and high-stakes applications.