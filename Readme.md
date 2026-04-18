# 🧬 PharmaRAG: Reliability & Governance Framework for Drug Safety QA

> A regulatory-aware RAG system that delivers evidence-grounded, citation-backed answers to drug safety questions — with agentic validation, hallucination detection, and MLOps-lite monitoring for pharma-grade trust.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LLM: Gemma 3](https://img.shields.io/badge/LLM-Gemma%203%2012B-orange?logo=google&logoColor=white)](https://ai.google.dev/gemma)
[![Ollama](https://img.shields.io/badge/Runtime-Ollama-black?logo=ollama&logoColor=white)](https://ollama.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---


## 🔍 The Problem

Healthcare and pharma professionals increasingly turn to LLMs for drug and clinical questions. But current systems suffer from critical shortcomings that make them **unsafe for regulated use**:

| Gap | Risk |
|-----|------|
| **Hallucinations** | LLMs fabricate drug interactions, dosages, or contraindications |
| **Weak citations** | Answers lack traceable evidence back to authoritative sources |
| **No governance** | Missing audit logs, monitoring, drift detection, and refusal policies |
| **Prototype-only research** | Most RAG-in-healthcare papers stop at accuracy — ignoring operational readiness |

## 💡 What This Project Does

PharmaRAG is a **prototype drug label & safety QA system** that answers questions about indications, contraindications, warnings, adverse reactions, dosing, and drug interactions — grounded in FDA-approved drug labels.

### Concrete Use Case

A clinician or analyst asks:

> *"What are the contraindications for ocrelizumab in patients with hepatitis B?"*


PharmaRAG returns:

- ✅ A **plain-language answer** grounded in retrieved evidence
- 📑 **Numbered citations** pointing to specific label sections and text snippets
- 📊 An **evidence table** with source, section, chunk ID, and retrieval score
- 🚦 A **confidence decision**: `Answer` · `Answer with Caution` · `Insufficient Evidence (Refused)`
- 📝 **System logs** for audit: latency, retrieval scores, groundedness signals, refusal reasons

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │  QUERY ROUTER   │  Agent 1: Maps question → label
            │  (Section       │  section type (indications,
            │   Classifier)   │  contraindications, warnings, etc.)
            └────────┬────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   HYBRID RETRIEVAL     │
        │  ┌──────┐  ┌────────┐ │
        │  │ BM25 │  │Vector  │ │  Weighted fusion:
        │  │(40%) │  │Search  │ │  60% semantic / 40% keyword
        │  └──┬───┘  └───┬────┘ │
        │     └─────┬─────┘     │
        │           ▼           │
        │      Re-Ranker        │
        └───────────┬───────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ EVIDENCE         │  Agent 2: Checks each answer
          │ VALIDATOR        │  sentence has a supporting
          │                  │  citation from retrieved chunks
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ REFUSAL GUARD    │  Agent 3: Abstains when evidence
          │                  │  confidence is below threshold
          └────────┬─────────┘
                   │
                   ▼
     ┌──────────────────────────┐
     │    STRUCTURED OUTPUT     │
     │  • Answer + Citations    │
     │  • Evidence Table        │
     │  • Confidence Level      │
     └────────────┬─────────────┘
                  │
                  ▼
     ┌──────────────────────────┐
     │   MONITORING & LOGGING   │
     │  • request_id, timestamp │
     │  • retrieval scores      │
     │  • groundedness score    │
     │  • latency (per-stage)   │
     │  • refusal flags         │
     └──────────────────────────┘
```

---

## ✨ Key Features
 
### 🤖 Agentic Safeguards
 
| Agent | Purpose |
|-------|---------|
| **Query Router** | Classifies incoming questions and routes to the most relevant label sections, improving retrieval precision |
| **Evidence Validator** | Verifies every answer sentence is supported by a cited chunk — flags unsupported claims |
| **Refusal Guard** | Triggers `Insufficient Evidence` when retrieval confidence is below threshold — the system knows when to say "I don't know" |
 
### 📊 Governance & Monitoring (MLOps-Lite)
 
Every request generates a structured audit log:
 
```json
{
  "request_id": "req_a1b2c3",
  "timestamp": "2025-04-15T10:32:00Z",
  "query": "What are the black box warnings for fingolimod?",
  "retrieved_docs": ["dailymed_fingolimod_chunk_42", "dailymed_fingolimod_chunk_17"],
  "top_k_scores": [0.91, 0.87, 0.73, 0.68, 0.61],
  "latency_ms": { "retrieval": 320, "generation": 2100, "validation": 180, "total": 2600 },
  "groundedness_score": 0.92,
  "confidence_level": "answer",
  "refusal": false
}
```
 
### 🔒 Regulatory Alignment
 
- **Data provenance**: Every answer traces back to specific FDA label sections
- **Refusal policy**: System refuses to answer rather than hallucinate
- **Local LLM**: Runs via Ollama — no data leaves your infrastructure
- **Audit trail**: Full logging for compliance review
 
---
 
## 📈 Evaluation & Metrics
 
### Metric Targets
 
| Category | Metric | Target | Description |
|----------|--------|--------|-------------|
| **Retrieval** | Recall@5 | ≥ 0.70 | Top-5 results include the correct label section |
| **Retrieval** | nDCG@5 | ≥ 0.60 | Best sections are ranked near the top |
| **Grounding** | Groundedness Rate | ≥ 85% | % of answer sentences supported by cited chunks |
| **Grounding** | Hallucination Rate | ≤ 10% | % of sentences not supported or contradicted by evidence |
| **Grounding** | Citation Precision | High | % of citations that truly support the claim |
| **System** | P95 Latency | ≤ 6–8s | End-to-end response time (local LLM) |
| **System** | Refusal Correctness | High | Refuses when evidence is weak; answers when evidence exists |
 
### Evaluation Plan
 
- **Test set**: 75–150 queries balanced across label sections (indications, contraindications, warnings, AEs, dosing, interactions)
- **Manual review**: 50 queries scored for groundedness and hallucination
- **Automated scoring**: Full retrieval metrics across the complete test set
- **Ablation study**: Metrics with and without agentic checks to quantify their impact
 
---

# 🛠 Tech Stack
 
| Component | Technology |
|-----------|-----------|
| **LLM (Primary)** | Gemma 3 12B Instruct via Ollama |
| **LLM (Fallback)** | Gemma 3 4B Instruct / Llama 3.2 3B |
| **Embeddings** | Sentence Transformers (all-MiniLM or domain-specific) |
| **Vector Store** | ChromaDB / FAISS |
| **Keyword Search** | BM25 (rank-bm25) |
| **Reranking** | Cross-encoder reranker |
| **Orchestration** | LangChain / custom pipeline |
| **Frontend** | Streamlit |
| **Backend** | FastAPI (optional) |
| **Monitoring** | Custom logging + JSON audit trail |
| **Language** | Python 3.10+ |
 
---
 
## 📂 Data Sources
 
| Source | Role | Format |
|--------|------|--------|
| [**FDA DailyMed**](https://dailymed.nlm.nih.gov/) | Primary evidence — Structured Product Labels (SPL) | XML/API |
| [**PubMed**](https://pubmed.ncbi.nlm.nih.gov/) | Secondary — supporting abstracts for context | API |
| [**ClinicalTrials.gov**](https://clinicaltrials.gov/) | Tertiary — trial-level summaries | API |
 
### Scope
 
Initial focus: **25–50 drugs** in a single therapeutic area (e.g., Multiple Sclerosis, Oncology, or Immunology) with potential expansion to a generalized top-50 drug set.
 
### Data Pipeline
 
```
FDA DailyMed API  →  Pull relevant SPLs  →  Parse XML sections  →  Chunk by section
                                                                          │
PubMed API        →  Curated abstracts   →  Parse & clean       →        ▼
                                                                   Hybrid Index
ClinicalTrials    →  Trial summaries     →  Parse & clean       →  (BM25 + Vector)
```
 
---

## 🚀 Getting Started
 
### Prerequisites
 
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- 16GB+ RAM recommended (for 12B model)
 
### Installation
 
```bash
# Clone the repository
git clone https://github.com/yourusername/pharma-rag.git
cd pharma-rag
 
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
 
# Install dependencies
pip install -r requirements.txt
 
# Pull the LLM model
ollama pull gemma3:12b
 
# (Optional) Pull fallback models
ollama pull gemma3:4b
ollama pull llama3.2:3b
```

## 💬 Usage
 
### Example Queries
 
**Patient-style:**
> "Can I take Tecfidera if I have a low white blood cell count?"
 
**HCP / Analyst-style:**
> "Summarize the black box warnings and contraindications for natalizumab, citing the relevant label sections."

### Sample Output
 
```
📋 Answer:
Natalizumab carries a black box warning for progressive multifocal
leukoencephalopathy (PML), a serious brain infection [1][2]. It is
contraindicated in patients who have or have had PML and in patients
with hypersensitivity to natalizumab [3].
 
📑 Citations:
[1] DailyMed — Natalizumab — Section: BOXED WARNING — Chunk ID: nat_bw_001
[2] DailyMed — Natalizumab — Section: WARNINGS AND PRECAUTIONS — Chunk ID: nat_wp_012
[3] DailyMed — Natalizumab — Section: CONTRAINDICATIONS — Chunk ID: nat_ci_003
 
🚦 Confidence: ██████████ ANSWER (Groundedness: 0.94)
```
 
---
 
## 📁 Project Structure
 
```
pharma-rag/
├── app/
│   ├── main.py                 # Streamlit UI entry point
│   ├── api.py                  # FastAPI backend (optional)
│   └── components/             # UI components
├── src/
│   ├── ingestion/
│   │   ├── dailymed.py         # DailyMed SPL fetcher & parser
│   │   ├── pubmed.py           # PubMed abstract fetcher
│   │   └── clinical_trials.py  # ClinicalTrials.gov fetcher
│   ├── indexing/
│   │   ├── chunker.py          # Section-aware chunking
│   │   ├── embedder.py         # Embedding generation
│   │   └── index_builder.py    # Hybrid index (BM25 + vector)
│   ├── retrieval/
│   │   ├── query_router.py     # Agent 1: Query → section classifier
│   │   ├── hybrid_search.py    # BM25 + semantic fusion
│   │   └── reranker.py         # Cross-encoder reranking
│   ├── generation/
│   │   ├── generator.py        # LLM answer generation
│   │   ├── evidence_validator.py  # Agent 2: Citation grounding check
│   │   └── refusal_guard.py    # Agent 3: Confidence-based refusal
│   ├── monitoring/
│   │   ├── logger.py           # Structured audit logging
│   │   ├── metrics.py          # Latency & score tracking
│   │   └── dashboard.py        # Monitoring dashboard
│   └── evaluation/
│       ├── test_set.py         # Test query management
│       ├── retrieval_eval.py   # Recall@k, nDCG@k scoring
│       ├── grounding_eval.py   # Groundedness & hallucination scoring
│       └── report_generator.py # Evaluation report + charts
├── data/
│   ├── raw/                    # Raw SPL XML, abstracts
│   ├── processed/              # Parsed & chunked documents
│   └── index/                  # Vector store & BM25 index
├── configs/
│   ├── model_config.yaml       # LLM and embedding settings
│   ├── retrieval_config.yaml   # Search weights, top-k, thresholds
│   └── monitoring_config.yaml  # Logging and alerting settings
├── evaluation/
│   ├── test_queries.json       # 75–150 evaluation queries
│   ├── ground_truth.json       # Expected answers & sections
│   └── results/                # Evaluation outputs & charts
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_analysis.ipynb
│   └── 03_evaluation_report.ipynb
├── tests/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```
## 🗺 Roadmap
 
- [x] Project design & framework specification
- [ ] Data ingestion pipeline (DailyMed SPL)
- [ ] Section-aware chunking & hybrid indexing
- [ ] Query router agent
- [ ] Base RAG generation with citations
- [ ] Evidence validator agent
- [ ] Refusal guard agent
- [ ] Monitoring & audit logging
- [ ] Evaluation framework & test set
- [ ] Streamlit UI
- [ ] Evaluation report & ablation study
- [ ] Demo video & documentation
 
---

## 🤝 Contributing
 
This is a Masters capstone project. Feedback, suggestions, and issues are welcome!
 
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

Results

1.
2.
3.
4.
5.
6.


