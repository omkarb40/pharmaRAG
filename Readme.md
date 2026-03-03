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

-
-
-
-
-

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