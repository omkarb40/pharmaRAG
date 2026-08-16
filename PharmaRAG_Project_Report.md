# PharmaRAG: A Safety-Governed Retrieval System for Drug-Label Question Answering

### Capstone Project Report

**Programme:** M.S. Data Science
**Author:** Omkar
**Supervisor:** [name]
**Date:** [submission date]

---

> **Note on this draft.** Placeholders appear in three forms. `[FIGURE n]` marks a visualisation to produce, with capture instructions in Appendix C. `[SCREENSHOT n]` marks an interface capture, likewise in Appendix C. `[RUN: block n]` marks a number I could not compute without access to the repository; the corresponding code is in Appendix B. Everything else is final.

---

## Executive Summary

This project asked whether a retrieval-augmented question-answering system for drug safety can be made trustworthy enough for a regulated setting, and, more importantly, whether that trustworthiness can be measured rather than asserted. The answer to the first question turned out to depend entirely on the second.

I built PharmaRAG, a fully local system that answers drug-safety questions over FDA Structured Product Labels for 28 Multiple Sclerosis medications. It retrieves evidence using a hybrid of lexical and semantic search, reranks that evidence with a cross-encoder, generates an answer with a locally hosted language model, and then subjects that answer to three governing agents that route the query, validate the answer sentence by sentence against the retrieved evidence, and decide whether the answer should be released at all. Every request produces an audit record.

The system works. Retrieval recall at rank five reaches 0.845, groundedness sits at 0.932, and the proportion of questions that should have been refused but were answered anyway falls to 0.111.

The finding that mattered most, however, came from a configuration I built only as a control. When I stripped out the three agents and measured the remaining pipeline, its groundedness was 0.945, slightly *higher* than the full system. Every answer it produced was well supported by evidence. It also answered every single question that should have been refused, including a request for a paediatric dose of a drug licensed only for adults. That answer was accurate, correctly cited, and grounded in a real FDA label. It was also the most dangerous output the system ever produced.

That result reframed the project. Groundedness, the metric the entire retrieval-augmented generation literature treats as the proxy for trustworthiness, turned out to be blind to the failure mode I most needed to catch. The remainder of the work went into building evaluation instruments that were not blind to it: a benchmark in which nearly a third of the questions are designed to be refused, a corrected way of computing groundedness that does not punish a system for abstaining correctly, and a decomposition of citation errors that separates a formatting mistake from a fabrication.

Along the way the project surfaced three problems I did not anticipate and which I regard as the most instructive parts of the work. A silent corpus corruption in which three drugs were indexed as the wrong formulation entirely, passing every automated check and caught only by manual annotation. A retrieval fusion configuration that looked like a hybrid system and was mathematically incapable of behaving like one. And a groundedness metric that was understating the system's own performance by 33 percentage points because of how it treated refusals.

---

## Table of Contents

1. Introduction and Motivation
2. Background and Related Systems
3. Scope, Requirements, and Success Criteria
4. System Architecture
5. Implementation and Development Phases
6. Corpus Construction and a Data Integrity Failure
7. Benchmark Construction and Annotation
8. Evaluation Methodology
9. Results
10. Failure Analysis
11. Engineering Reflections
12. Limitations
13. Future Work
14. Conclusion
15. References
- Appendix A: Repository Structure and Reproduction
- Appendix B: Code Blocks for Outstanding Numbers
- Appendix C: Figure and Screenshot Inventory
- Appendix D: Full Query Benchmark

---

## 1. Introduction and Motivation

A clinician wants to know whether a multiple sclerosis therapy is contraindicated in a patient with active hepatitis B. A pharmacist needs the dose adjustment for renal impairment. A patient reads about a drug online and wants to know what the warnings actually say. All three questions have authoritative answers, and all three answers live in the same place: the FDA Structured Product Label, a regulated document with a fixed section structure that every approved drug in the United States must carry.

The problem is that these documents are long, dense, and written for a professional reader. Finding the relevant paragraph in a forty-page label is slow. This is exactly the shape of problem retrieval-augmented generation was designed for, and a number of groups have built systems that do it.

What concerned me from the start of this project was not whether such a system could be built, but how anyone would know whether it was safe to use. The systems I reviewed all made a version of the same claim: because answers are constrained to retrieved label content, hallucination is minimised. That claim is architecturally reasonable and almost never measured. In one of the closest comparable systems, the authors implement a refusal mechanism, describe it in the methods section, and never test whether it fires correctly. They also state plainly that inter-annotator agreement was not assessed and that quantitative hallucination rates were not computed.

This is not a criticism of that work specifically. It reflects a gap in the field. The benchmarks that define progress in medical retrieval-augmented generation score multiple-choice accuracy, and multiple-choice accuracy has no way to express the concept of a question that should not have been answered. A system that answers everything scores identically to a system that abstains appropriately, because abstention is not representable in the format.

For a drug-safety system, that is precisely backwards. The dangerous failure is not getting a question wrong. It is confidently answering a question that should have been declined.

The project therefore has two objectives, and they are ordered:

**Primary.** Build an evaluation framework that can measure whether a drug-safety RAG system abstains when it should, grounds its claims when it answers, and attributes those claims to the correct evidence, and construct the human-annotated benchmark such a framework requires.

**Secondary.** Build a system worth evaluating, comprising a retrieval pipeline over FDA labels and an agentic governance layer that makes explicit release decisions.

The secondary objective came first chronologically. The primary objective is what the project turned out to be about.

---

## 2. Background and Related Systems

### 2.1 Retrieval-augmented generation

Retrieval-augmented generation, introduced by Lewis et al. (2020), addresses a structural weakness of language models: their knowledge is fixed at training time, opaque in provenance, and unreliable at the tail of the distribution. Rather than relying on parametric memory, a RAG system retrieves relevant passages from an external corpus at query time and conditions generation on them. The retrieved passages can be cited, which gives every claim a traceable source.

For drug safety this is close to a necessary architecture rather than an optional one. Label content changes as new safety information is added, the source document is legally authoritative, and any answer a clinician acts on must be verifiable against that document.

### 2.2 Systems that answer questions about drug labels

Several systems target this task directly, and reviewing them shaped both what I built and what I chose to measure.

Koppula et al. (2025) constrain GPT-3.5 to sections parsed from an uploaded label PDF, evaluate by semantic similarity against manually extracted ground truth, and report a refusal mechanism that fires when no relevant section is retrieved. Their limitations section is unusually candid and is worth quoting in substance: inter-annotator agreement was not formally measured, and quantitative hallucination rates were not measured. Their system is architecturally similar to a subset of mine. The difference is entirely in what gets measured.

Nishisako et al. (2025) are closer to the evaluation posture I eventually adopted. They compare six chatbot configurations for cancer information, score every response by review from two experienced clinicians, and report a trade-off directly: as the rate of medically harmful content falls from roughly 40 percent to between 0 and 6 percent, the response rate falls from 100 percent to between 36 and 81 percent. Crucially, they treat non-response as a desirable property rather than a failure. This was the first paper I read that scored abstention as anything other than a loss, and it directly influenced my decision to build refusal cases into the benchmark.

Concurrent with this project, Wang et al. (2026) released DrugClaw and DrugAudit, a multi-agent drug-information system paired with a 3,772-item benchmark that scores source authority and citation faithfulness and rewards calibrated abstention. Their scale exceeds mine by a factor of forty-four. Two differences justify the present work. Their questions and gold answers are generated by a language model and scored by language-model judges, a design they adopt deliberately and identify as warranting expert re-annotation; mine are written and scored by humans throughout. And their refusal items are defined by evidentiary absence, meaning the system is correct to abstain because the record is empty. My most difficult refusal cases are the opposite: the evidence is present, real, and retrievable, and the correct behaviour is still to decline.

### 2.3 Self-reflective retrieval architectures

The agentic layer in this project descends from work that gives a RAG pipeline the ability to inspect its own retrieval. Self-RAG (Asai et al., 2024) trains a model to emit reflection tokens judging retrieval necessity and passage relevance. Corrective RAG adds an evaluator that classifies retrieval quality and takes remedial action when it is poor. My Query Router, Evidence Validator, and Refusal Guard follow this pattern with one shift of emphasis. Self-reflective RAG uses validation to improve an answer. I use it to decide whether to release one.

### 2.4 Evaluating faithfulness

Frameworks for scoring RAG output converge on faithfulness, meaning the proportion of generated claims supported by retrieved context. FActScore (Min et al., 2023) establishes the decomposition of a generation into individually verifiable atomic units, which is the principle my per-sentence groundedness measure implements. RAGAS and ARES operationalise faithfulness and context relevance at the framework level.

None of these frameworks, as far as I can determine, scores refusal against a labelled ground truth of which questions should have been refused. That gap is the space this project occupies.

---

## 3. Scope, Requirements, and Success Criteria

### 3.1 Scope decisions and their rationale

**Therapeutic area: Multiple Sclerosis, 28 drugs.** A single therapeutic area keeps the corpus small enough to annotate by hand, which was a hard requirement once I decided that gold annotations had to be human-generated. MS specifically offers a useful property: the drug class includes agents with serious boxed warnings, complex dosing, significant pregnancy restrictions, and adult-only indications, which supplies natural material for safety-critical test queries.

**Evidence source: DailyMed SPL only.** I scoped out PubMed abstracts and ClinicalTrials.gov summaries, both of which appeared in the original project proposal. The reason is that mixing a regulatory primary source with a literature secondary source makes citation integrity much harder to define, and citation integrity was the point. A claim traceable to an FDA label is verifiable in a way that a claim traceable to an abstract of a trial is not.

**Local generation only.** All inference runs locally through Ollama. This is a governance requirement rather than a performance preference. In any realistic deployment the query itself may carry clinical context, and sending it to a third-party API would undermine the regulated-setting framing the whole project rests on. It also makes every result exactly reproducible.

**Text only.** Labels contain tables and occasional figures. Parsing these reliably is a project in itself and would have consumed time better spent on evaluation.

### 3.2 Requirements

| ID | Requirement | Status |
|---|---|---|
| F1 | Answer drug-safety questions from indications, contraindications, warnings, adverse reactions, dosing, interactions, and specific-populations sections | Met |
| F2 | Attach numbered citations resolving to a specific label section and text chunk | Met |
| F3 | Emit a three-tier release decision: answer, answer with caution, or refuse | Met |
| F4 | Abstain when retrieved evidence is insufficient | Met |
| F5 | Log every request with retrieval trace, scores, latency, and decision | Met |
| F6 | Run entirely on local infrastructure | Met |
| N1 | End-to-end latency under 20 seconds on consumer hardware | Met (15.3 s mean) |
| N2 | Deterministic and reproducible outputs | Met (temperature 0) |
| N3 | Evaluation reproducible from a single command | Met |

### 3.3 Success criteria, as proposed and as achieved

The criteria below were set at proposal stage, before any results existed. I report them against outcomes without adjustment.

| Criterion | Target | Achieved | Verdict |
|---|---|---|---|
| Recall@5 on label-section retrieval | ≥ 0.70 | 0.845 | Exceeded |
| nDCG@5 | ≥ 0.60 | 0.978 | Exceeded |
| Groundedness (manual sample) | ≥ 0.85 | 0.932 | Met |
| Hallucination rate (manual sample) | ≤ 0.10 | 0.082 | Met |
| P95 latency, local | ≤ 6 to 8 s | [RUN: block 4] | See note |
| Refusal correctness | Not quantified at proposal | 0.777 | New metric |

Two comments on this table. The latency target was set before I added cross-encoder reranking, and reranking pushed retrieval cost up by an order of magnitude. Mean end-to-end latency is 15.3 seconds, dominated by generation rather than retrieval. I regard the target as having been set wrongly rather than the system as having failed it, and I discuss the trade-off in Section 9.5.

More significantly, the proposal contained no target for refusal correctness because at proposal stage I had not understood that refusal was the central problem. That absence is itself a finding about how easy it is to specify a drug-safety system without specifying the property that makes it safe.

---

## 4. System Architecture

`[FIGURE 1]` Architecture diagram.

The pipeline has six stages. A query arrives, the Query Router classifies it to a label section, hybrid retrieval returns candidates from that section, a cross-encoder reranks them, a local language model generates an answer from the top five, and the Evidence Validator and Refusal Guard decide whether that answer is released, released with a caution, or withheld. An audit record is written regardless of outcome.

### 4.1 Ingestion and chunking

FDA Structured Product Labels are XML documents in which each section carries a LOINC code identifying its regulatory purpose. Section 34067-9 is indications and usage, 34070-3 is contraindications, 43685-7 is warnings and precautions, and so on. This structure is the single most useful property of the corpus and most systems discard it by flattening the document to text.

I parse by LOINC code and preserve section identity as chunk metadata. Chunking is section-aware, meaning a chunk is never permitted to span a section boundary even when that produces a short chunk. Target size is 500 tokens with 50 tokens of overlap within a section. The resulting index holds 723 chunks across 28 drugs.

`[RUN: block 1]` produces the per-drug and per-section chunk distribution table.

### 4.2 Embeddings and index

Chunks are embedded with PubMedBERT (`neuml/pubmedbert-base-embeddings`) into 768 dimensions and stored in ChromaDB under cosine similarity.

The choice of a biomedical sentence-embedding model over a general-purpose one matters here. Label text is saturated with drug names, adverse-event terminology, and dosing constructions that general-purpose embeddings represent poorly. One trap worth recording: several widely cited biomedical encoders are token-level models trained for named-entity recognition and are not designed to produce sentence-level similarity. Using one of those would have degraded retrieval silently, which given the RRF experience described in Section 10.2 is a failure mode I now take seriously.

### 4.3 Retrieval

Retrieval is hybrid. BM25 supplies lexical matching, which matters because exact tokens carry disproportionate signal in this domain. A query about hepatotoxicity should match the label text containing the word hepatotoxicity, and dense retrieval alone is comparatively weak at guaranteeing that. Dense retrieval supplies robustness to paraphrase, which patient-style queries need, since a patient asks whether a drug is safe in pregnancy rather than asking about use in specific populations.

The two ranked lists are fused by weighted Reciprocal Rank Fusion with k of 60 and weights of 0.6 semantic to 0.4 lexical:

```
RRF(d) = w_s / (k + rank_s(d))  +  w_l / (k + rank_l(d))
```

Section 10.2 documents what went wrong with this configuration and how I found it.

### 4.4 Reranking

The top twenty fused candidates are rescored by a cross-encoder (`ms-marco-MiniLM-L-6-v2`) and truncated to the top five, which becomes the generation context. A cross-encoder scores query and candidate jointly rather than comparing independent embeddings, which is more accurate and much slower. Applying it to twenty candidates rather than to the whole corpus is what makes the cost tolerable.

### 4.5 Generation

Gemma 3 12B Instruct, served locally by Ollama, at temperature 0. The prompt supplies the five retrieved chunks with their section labels and instructs the model to answer only from that content and to cite chunk indices inline.

`[SCREENSHOT 1]` Ollama running locally with the model loaded.

### 4.6 The agentic layer

Three agents govern the pipeline, each addressing a different failure.

**Query Router.** Classifies the incoming query to a probable label section and restricts retrieval to chunks carrying that section label. This exploits the regulated structure directly: the answer to a contraindications question is, by construction, in the contraindications section. Section 9.5 reports that the router is worth 8.6 points of Recall@5, making it the cheapest accuracy gain in the system.

**Evidence Validator.** Splits the generated answer into sentences, embeds each, and compares it against the embeddings of the five supplied chunks. A sentence counts as grounded if its maximum similarity to any supplied chunk reaches 0.35. The response-level groundedness score is the fraction of sentences that clear the threshold.

**Refusal Guard.** Combines three signals into a scalar confidence:

```
C = 0.25 * retrieval_score  +  0.55 * groundedness  +  0.20 * evidence_count
```

and maps it to a decision: refuse below 0.45, caution below 0.65, otherwise answer.

The 0.55 weight on groundedness reflects a design assumption I held at the time and which the results later undermined. I assumed that how well an answer is supported by evidence is the dominant indicator of whether it is safe to release. Section 10.1 documents where that assumption fails.

### 4.7 Audit logging

Every request writes a JSONL record containing request identifier, timestamp, routed section, retrieved chunk identifiers with scores, per-stage latency, per-sentence groundedness, computed confidence, and the guard's decision with reason.

`[SCREENSHOT 2]` A formatted audit log record.

---

## 5. Implementation and Development Phases

The project ran in five phases, each scoped to a demonstrable milestone. I preserve the demo scripts from each phase in `demos/` rather than deleting them, because they document how the system's behaviour changed.

### Phase 1: Ingestion and baseline retrieval

Built the DailyMed API client, SPL XML parser, LOINC section mapping, chunker, and ChromaDB index. Demo milestone was a semantic search returning relevant label passages for a plain-language query.

The main difficulty was that SPL XML is considerably less regular than the specification suggests. Section nesting varies between manufacturers, some sections appear twice with different content granularity (a highlights summary and a full section), and section codes are occasionally absent.

### Phase 2: Generation and citation

Added Ollama integration, prompt construction, and citation parsing. Demo milestone was a cited answer to a contraindications query.

Citation parsing proved fiddly. The model would sometimes cite in prose rather than by index, or cite a range, or cite a chunk it had not used. This is the origin of the citation misalignment problem quantified in Section 9.3.

### Phase 3: Hybrid retrieval and reranking

Added the BM25 index, RRF fusion, and cross-encoder reranking. Demo milestone was a side-by-side comparison of dense-only against hybrid-plus-rerank on a query where lexical matching should matter.

### Phase 4: The agentic layer

Added all three agents and the three-tier decision. Demo milestone was the system correctly refusing a question about a drug outside the corpus.

`[SCREENSHOT 3]` The system answering a well-supported question.
`[SCREENSHOT 4]` The system refusing an out-of-scope question.
`[SCREENSHOT 5]` The system returning answer with caution.

### Phase 5: Benchmark, evaluation harness, and monitoring

Built the 85-query benchmark, annotated gold chunks by hand, implemented the metric suite, ran the five-configuration ablation, and built the monitoring dashboard over the audit logs.

`[SCREENSHOT 6]` Monitoring dashboard.

This phase consumed more time than the four before it combined. That distribution is itself worth reporting: building the system took weeks, and establishing whether it could be trusted took considerably longer.

---

## 6. Corpus Construction and a Data Integrity Failure

### 6.1 Construction

Drugs were selected to span the MS therapeutic landscape: injectable interferons and glatiramer acetate, oral agents including fingolimod, siponimod, teriflunomide and dimethyl fumarate, and monoclonal antibodies including natalizumab, ocrelizumab, ofatumumab and alemtuzumab. Labels were pulled from the DailyMed API, parsed by LOINC section, chunked, embedded, and indexed.

`[RUN: block 1]` produces the corpus composition table for this section.

### 6.2 The failure

While annotating gold chunks for a dosing query about Kesimpta, I could not find dosing content matching any MS regimen I recognised. The retrieved chunk described an intravenous infusion schedule for chronic lymphocytic leukaemia.

Kesimpta is ofatumumab, given subcutaneously for MS. ARZERRA is also ofatumumab, given intravenously for CLL. The DailyMed name search for the active ingredient had returned the oncology label as the first hit, and my ingestion pipeline had taken it.

The same failure had occurred for two more drugs:

| MS product | Active ingredient | Wrongly ingested label | Indication of wrong label |
|---|---|---|---|
| Kesimpta | ofatumumab | ARZERRA | Chronic lymphocytic leukaemia |
| Mavenclad | cladribine | Cladribine Injection | Hairy cell leukaemia |
| Lemtrada | alemtuzumab | CAMPATH | B-cell chronic lymphocytic leukaemia |

Three of twenty-eight drugs, roughly eleven percent of the corpus, were labels for the correct molecule and the wrong product.

### 6.3 Why nothing caught it

This is the part I consider most instructive. Every automated check passed. The documents were valid FDA SPL XML. They parsed cleanly. LOINC sections resolved. Chunking produced sensible chunk sizes. Embeddings generated without error. Retrieval returned confident, high-similarity results. A schema validator, a null check, a chunk-count assertion, and a retrieval smoke test would all have reported success.

The failure was not a malformed document. It was a well-formed document about the wrong thing, and no structural check can detect that, because structurally there is nothing wrong.

It was caught by a human reading the retrieved text and noticing that the dose did not correspond to any regimen the drug is given under.

### 6.4 Remediation

I replaced name-based resolution with formulation-specific SetID pinning, so each drug maps to an explicit immutable document identifier rather than to whatever the search endpoint returns first. Re-ingestion raised the corpus from 698 to 723 chunks. All affected queries were re-annotated.

### 6.5 What I take from it

Retrieval systems in regulated domains need provenance validation performed by a person who understands the domain, because the dangerous corpus errors are semantic rather than structural. I would now treat any ingestion pipeline that resolves documents by name search as unsafe by default.

The failure also had a second-order consequence I did not expect. Because it was caught during gold annotation, the manual annotation process was not merely an evaluation cost. It functioned as a data-quality control. Had I generated gold annotations with a language model, as several published benchmarks do, the model would have annotated the oncology content without complaint and the corruption would have propagated into the evaluation itself.

---

## 7. Benchmark Construction and Annotation

### 7.1 Design

The benchmark holds 85 queries. Fifty-eight are answerable. Twenty-seven are labelled should-refuse. That proportion is deliberate: a benchmark composed only of answerable questions cannot measure the property I most needed to measure.

**Table 7.1: Query composition**

| Category | Label | Count | Purpose |
|---|---|---|---|
| contraindications | answer | `[RUN: block 2]` | Core safety section |
| adverse_reactions | answer | `[RUN: block 2]` | Core safety section |
| warnings | answer | `[RUN: block 2]` | Core safety section |
| dosing | answer | `[RUN: block 2]` | Structured numeric content |
| indications | answer | `[RUN: block 2]` | Core section |
| interactions | answer | `[RUN: block 2]` | Cross-section reasoning |
| populations | answer | `[RUN: block 2]` | Pregnancy, hepatic, renal, geriatric |
| patient_style | answer | `[RUN: block 2]` | Lay paraphrase |
| multi_drug | answer | `[RUN: block 2]` | Comparative, multi-entity |
| near_scope | refuse | 12 | Real drugs absent from corpus |
| out_of_scope | refuse | 10 | Outside the label-QA task |
| adversarial | refuse | 5 | Unsafe despite retrievable evidence |
| **Total** | | **85** | |

### 7.2 The three refusal regimes

These test different mechanisms and are analysed separately throughout.

**Near-scope** queries name real drugs genuinely absent from the corpus, for instance a question about a cardiology drug. Refusing them tests whether the system detects evidentiary absence.

**Out-of-scope** queries fall outside label question-answering entirely. Refusing them tests scope discipline.

**Adversarial** queries are the important class and the hardest to construct. Each asks something the system must decline even though the corpus contains real, relevant, retrievable evidence. The clearest example is a paediatric dosing request for a drug indicated only in adults. The label contains extensive dosing information. Retrieval will find it. The model will ground an answer in it perfectly. And the answer is unsafe, because the drug is not indicated in that population and the dose does not exist.

No evidence-availability signal can catch this, because the evidence is available. This class is, as far as I can tell, absent from published drug-QA benchmarks, whose refusal items are defined by missing records.

### 7.3 Annotation protocol

Every answerable query carries a gold evidence chunk: the passage a human annotator identified as containing the answer, recorded as a verbatim substring of a real corpus chunk so that retrieval can be verified by exact substring match rather than by judgement.

No gold chunk, gold answer, or query was produced by a language model. I want to state this plainly because it was a costly decision. Generating the benchmark with a model would have taken hours instead of weeks and would have allowed a much larger query set. I rejected it because evaluating a language model against targets produced by a language model introduces a circularity a citation-integrity benchmark cannot tolerate, and because, as Section 6.5 notes, the manual process caught a corpus corruption that an automated process would have propagated.

Refuse-queries carry a null gold and are scored on the decision alone.

### 7.4 Inter-annotator agreement

Five raters, four without domain expertise including myself and one domain expert, independently annotated a stratified twenty-query subset under two tasks.

**Table 7.2: Inter-annotator agreement**

| Task | Statistic | Value |
|---|---|---|
| Best-single-chunk selection | Fleiss κ | 0.076 |
| Gold-chunk acceptance | Mean raters accepting | 3.9 / 5 |
| Gold-chunk acceptance, domain expert | Acceptance rate | 80% |
| Gold-chunk acceptance, per rater | Acceptance rate | 80 / 80 / 100 / 50 / 80 % |

These two numbers must be read together and I report both rather than the flattering one.

Agreement on selecting the single best chunk is near chance. The reason is structural rather than a defect in the annotation guidelines. FDA labels restate the same safety information in several places: once in the highlights summary, once in the full warnings section, and sometimes again under specific populations. Asking five people to choose the single best among near-duplicates measures an arbitrary preference.

Agreement on whether a given gold chunk is a valid answer is high, at 3.9 of 5. This is the metric-aligned quantity, because Gold-chunk@5 requires the gold chunk to appear anywhere in the top five, not to be ranked first. The low kappa therefore does not undermine the retrieval metric, though it does bound any future claim requiring rank-one agreement.

One rater accepted only half the gold chunks. Their rejections are scattered across categories with no clustering, which is inconsistent with a systematic weakness in any annotation class and consistent with an individually strict threshold. I report the outlier rather than dropping it.

The finding I would highlight is that non-domain raters performed near chance on selection while the domain expert did not diverge markedly on acceptance. Choosing the best evidence in a regulatory drug label appears to require clinical literacy in a way that validating proposed evidence does not.

---

## 8. Evaluation Methodology

### 8.1 Retrieval metrics

**Recall@5** is the proportion of answerable queries for which at least one chunk from the correct section appears in the top five.

**Gold-chunk@5** is the proportion for which the specific human-annotated gold chunk appears in the top five. This is the stricter measure. Recall@5 can be satisfied by any plausibly relevant chunk from the right section, whereas Gold-chunk@5 requires the passage a human identified as containing the answer. I report both because the gap between them is diagnostic: a wide gap indicates a system retrieving topically correct but non-answering evidence.

**nDCG@5** measures whether relevant evidence is ranked near the top rather than merely present.

### 8.2 Generation metrics

**Per-sentence groundedness.** For an answer split into sentences and evidence chunks E, with threshold 0.35:

```
g(a) = (1/m) * SUM_i  1[ max over e in E of cos(embed(s_i), embed(e)) >= 0.35 ]
```

**Hallucination rate** is one minus groundedness over factual sentences.

**Citation precision** is the proportion of cited claims for which the cited index supports the claim.

### 8.3 A measurement error I made and corrected

My first groundedness implementation averaged over every query in the benchmark. On the baseline configuration it returned 0.41, which was far below the proposal target of 0.85 and did not match what I saw reading the outputs, which looked well grounded.

The cause was that a correct refusal contains no sentences grounded in evidence, so it scores zero. The estimator was assigning zero groundedness to the system's best behaviour, and was doing so more heavily the better the refusal logic became.

Restricting the average to queries the system actually answered gives 0.74 on the same outputs, a difference of 33 percentage points on identical system behaviour:

```
g_bar = (1 / |A|) * SUM over q in A of g(a_q)      where A = queries answered
```

I record this because it has a consequence beyond my own bug. Any abstention-capable RAG system evaluated under the naive estimator faces a perverse incentive: improving refusal lowers reported groundedness, so a developer optimising a single headline number is pushed toward answering more often. Reporting groundedness over answered queries alongside refusal accuracy over all queries removes the incentive.

### 8.4 Three-axis faithfulness

Citation faithfulness decomposes into three independent axes, and conflating them misattributes failures.

1. **Retrieval.** Was the supporting evidence retrieved at all? Measured by Recall@5 and Gold-chunk@5.
2. **Groundedness.** Is the claim supported by the evidence supplied? Measured by g(a).
3. **Attribution.** Does the citation index point to the chunk that actually supports the claim? Measured by citation precision.

A claim can pass the first two and fail the third: right evidence retrieved, claim correctly derived, wrong number printed beside it. That is a display defect, not a fabrication, and a single conflated metric treats them identically.

### 8.5 Safety metrics

**Unsafe emission rate** is the proportion of should-refuse queries for which the system emitted an answer.

**Refusal accuracy** is the proportion of all queries whose decision matched the label, over both directions. It is computed over the full benchmark because a system that refuses everything achieves a perfect unsafe-emission rate and is useless.

### 8.6 Statistical protocol

Adjacent configurations are compared with McNemar's test on paired per-query binary outcomes, appropriate for this paired binary design. I report discordant-pair counts alongside p-values so power is visible.

Refusal-subset comparisons are underpowered. With 27 refuse queries and only 5 adversarial, differences of one or two items are not resolvable and I do not claim significance for them.

---

## 9. Results

All results are from deterministic runs at temperature 0 and reproduce exactly on re-execution.

### 9.1 Pipeline ablation

**Table 9.1: Five-configuration ablation.** Groundedness and hallucination computed over answered queries only, per Section 8.3.

| Configuration | Recall@5 | Gold-chunk@5 | nDCG@5 | Groundedness | Hallucination | Refusal acc. | Unsafe emission |
|---|---|---|---|---|---|---|---|
| N: no agents | 0.586 | 0.328 | 0.894 | 0.945 | 0.060 | 0.682 | 1.000 |
| Baseline: hybrid | 0.707 | 0.448 | 0.965 | 0.960 | 0.049 | 0.706 | 0.185 |
| A: query expansion | 0.724 | 0.466 | 0.951 | 0.930 | 0.070 | 0.729 | 0.222 |
| B: context embeddings | 0.741 | 0.431 | 0.958 | 0.939 | 0.061 | 0.741 | 0.148 |
| **C: reranking (final)** | **0.845** | **0.603** | **0.978** | 0.932 | 0.082 | **0.777** | **0.111** |

Retrieval improves monotonically, with reranking contributing the largest single increment. Gold-chunk@5 rises from 0.328 to 0.603, a wider spread than Recall@5, which tells me reranking recovers the specific answering passage rather than merely topically adjacent text. Refusal accuracy climbs from 0.682 to 0.777.

Groundedness does not follow this pattern at all. It moves within a three-point band with no relationship to any pipeline change, and Configuration C, best on every other metric, is not best on groundedness. Query expansion degrades groundedness while improving recall, which I read as the expansion admitting loosely related evidence the generator then draws on.

`[FIGURE 2]` Per-category Recall@5 across configurations.

### 9.2 The safety result

**Table 9.2: Safety behaviour**

| Metric | Config N | Baseline | Config C |
|---|---|---|---|
| Unsafe emission rate (27 refuse queries) | 1.000 (27/27) | 0.185 (5/27) | 0.111 (3/27) |
| Adversarial emission (5 queries) | 5/5 | 3/5 | 2/5 |
| Groundedness on answered queries | 0.945 | 0.960 | 0.932 |

`[FIGURE 3]` Groundedness against unsafe emission, N versus C.

I want to be careful about what this table does and does not show, because the obvious reading of it is wrong.

Configuration N has no Refusal Guard, so it has no mechanism by which it could abstain. Its unsafe emission rate of 1.000 is a consequence of its architecture, not a measurement of its behaviour. Reporting an 89 percent reduction from N to C would be describing arithmetic as though it were a treatment effect, and I do not claim it. The Baseline to C difference, five unsafe emissions against three on 27 items, is not statistically resolvable and I do not claim that either.

What the table does show is a dissociation, and that is the substantive result.

Configuration N is as grounded as the full system. Every answer it gives is well supported by the evidence it retrieved. It also answers every question that should have been refused, including all five adversarial ones. A system can therefore be maximally grounded and maximally unsafe at the same time, because these are different properties and no amount of the first implies the second.

The consequence for evaluation practice is direct. A faithfulness metric applied to Configuration N returns an excellent score. That score is correct and certifies nothing. The unsafe answers are not hallucinations. They are accurate, well cited, evidence-grounded statements made in reply to questions that should not have been answered. No metric computed over the relationship between answer and evidence can detect this, because that relationship is fine. The defect is in the decision to answer.

**Table 9.3: Decision correctness by scope, Configuration C**

| Scope | Correct | Rate |
|---|---|---|
| In scope, answered correctly | 42 / 58 | 0.72 |
| Near scope, refused | 11 / 12 | 0.92 |
| Out of scope, refused | 10 / 10 | 1.00 |
| Adversarial, refused | 3 / 5 | 0.60 |

The gradient here runs exactly as the mechanism predicts. Out-of-scope refusal is perfect and near-scope refusal near-perfect, because in both cases the corpus holds no supporting evidence and a groundedness-driven confidence score falls accordingly. Adversarial refusal is the worst result in the entire benchmark, because the evidence is present. A 40 percent failure rate on the safety-critical class is a real gap, and I discuss its mechanism in Section 10.1.

### 9.3 Citation faithfulness

**Table 9.4: Decomposition of citation failures, Configuration C**

| Failure type | Count | Share |
|---|---|---|
| Misalignment: claim supported by a retrieved chunk, wrong index cited | 23 | 85% |
| Genuinely unsupported: no retrieved chunk supports the claim | 4 | 15% |
| **Total** | **27** | 100% |

Naive citation precision is 0.72, and that number conflates two very different defects. Of 27 failures, 23 are attribution errors where the claim is correct, the supporting evidence was retrieved and supplied, and the printed index points elsewhere within the top five. Four are genuine unsupported assertions. Across 96 cited claims the true hallucination rate is therefore about 4 percent.

The operational difference matters. A misaligned citation is a traceability defect a reviewer can correct against the source document in seconds. A fabrication cannot be corrected, because there is nothing to correct it against.

One caveat: support is assessed against a 200-character snippet of the cited chunk, so support present in the chunk but outside the snippet window is scored as a failure. The 0.72 figure is a lower bound and the 23-item misalignment count an upper bound. The scorer also shares an embedding model with the Evidence Validator, so the two measures are not fully independent.

### 9.4 Retrieval components

**Table 9.5: Component ladder**, answerable queries, router held constant

| Configuration | Recall@5 | Gold-chunk@5 | nDCG@5 |
|---|---|---|---|
| BM25 only | 0.5345 | 0.2931 | 0.9155 |
| Dense only | 0.7759 | 0.5172 | 0.9602 |
| Hybrid RRF | 0.7069 | 0.4483 | 0.9626 |
| Hybrid + rerank | **0.8448** | **0.6034** | **0.9770** |

McNemar on adjacent pairs, Recall@5: BM25 to Dense, +18/−4, p = 0.0043. Dense to Hybrid, +2/−6, p = 0.289, not significant. Hybrid to Hybrid plus rerank, +8/−0, p = 0.0078. On Gold-chunk@5, Hybrid to rerank gives +11/−2, p = 0.0225.

BM25 appears as the standard information-retrieval reference floor, not as an external system comparison.

Two observations. Reranking is the only component whose contribution is both large and statistically clean, improving eight queries and degrading none. And hybrid fusion does not improve on dense retrieval; the transition is non-significant and points the wrong way. Section 10.2 explains why.

`[FIGURE 4]` Semantic weight sweep.

### 9.5 Router contribution and latency

Disabling the Query Router while holding everything else constant drops Recall@5 from 0.707 to 0.621. The router is worth 8.6 points and is the cheapest accuracy gain in the system, because it exploits document structure that already exists rather than learning anything.

**Table 9.6: Mean latency by stage, Configuration C**

| Stage | Mean latency |
|---|---|
| Query routing | 2.15 s |
| Retrieval and reranking | 2.27 s |
| Generation | 10.3 s |
| Validation | 0.55 s |
| **End to end** | **15.3 s** |

`[FIGURE 5]` Latency breakdown.
`[RUN: block 4]` for P95 and full distribution.

Generation dominates. Reranking raises retrieval cost roughly elevenfold over the baseline's 82 milliseconds, but at 2.27 seconds it remains about a fifth of generation cost, so the accuracy gain in Section 9.4 is bought at a latency price that barely registers against the dominant term. The Evidence Validator, which does the safety-critical work, costs 0.55 seconds. Safety, in this architecture, is cheap.

---

## 10. Failure Analysis

Three failure modes are worth analysing in detail. All three were surfaced by the evaluation framework, and I would not have found any of them by reading outputs.

### 10.1 Adversarial queries: well-grounded answers that should not exist

Configuration C refuses only three of five adversarial queries. Tracing one of the failures through the pipeline shows why, and the answer is uncomfortable, because nothing malfunctioned.

Take a request for a paediatric dose of a drug licensed only in adults. Retrieval works correctly and returns real dosing content for that drug. Generation works correctly and produces an answer well supported by that content. The Evidence Validator works correctly and computes high per-sentence groundedness, because the sentences genuinely are grounded. The Refusal Guard, which weights groundedness at 0.55, computes high confidence and releases the answer.

Every component did what it was designed to do. The system produced a well-grounded, correctly cited, factually accurate answer to a question whose correct response was refusal.

This is not a tuning problem. Lowering the confidence threshold would trade this failure for a large number of incorrect refusals on legitimate queries, because the confidence signal genuinely is high here and genuinely should be. The problem is that abstention driven by evidence quality is silent when the evidence is fine and the *question* is the problem.

The architectural implication is that safety gating needs a signal orthogonal to evidence quality: a classifier operating on the query rather than the retrieval, sitting before or alongside the evidence pipeline. I document this as future work and have not built it.

The evaluation implication is more general. A benchmark whose refusal items are defined by missing evidence cannot detect this class of failure at all, because in every such item evidentiary absence and correct refusal coincide, so a system that refuses on absence alone appears to solve the problem. Separating the two requires refusal items where the evidence is present, which is why the adversarial class exists in this benchmark.

I regard this as the most transferable finding in the project.

### 10.2 Hybrid retrieval that was not hybrid

Table 9.5 shows hybrid fusion performing worse than dense retrieval alone, which should not happen. Adding a second retrieval signal should not make things worse. I initially assumed a bug in the BM25 index.

Sweeping the semantic fusion weight showed something else:

**Table 10.1: Recall@5 against semantic fusion weight**

| Semantic weight | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| Recall@5 | 0.655 | 0.707 | 0.707 | 0.724 | 0.741 | 0.776 | 0.776 |

Recall rises monotonically with semantic weight and plateaus exactly at the dense-only value. The lexical component was contributing nothing positive at any setting.

Inspecting the fused scores gave the mechanism. With k of 60 and weights of 0.6 to 0.4, the RRF score of the worst-ranked dense candidate exceeds the score of the best-ranked candidate that only BM25 returned. Concretely, a document ranked twentieth by dense retrieval scores 0.6/(60+20) = 0.0075, while a document ranked first by BM25 and absent from the dense list scores 0.4/(60+1) = 0.0066. The lexical-only document loses.

BM25 could therefore reorder documents the dense retriever had already returned, but could never introduce one the dense retriever had missed. What I had described in my Phase 3 demo as a hybrid retrieval system was structurally dense-only retrieval with a lexical re-scoring term.

This is confirmed downstream: hybrid plus rerank and dense plus rerank produce identical top-five sets on all 58 answerable queries, with zero discordant pairs.

I have deliberately not re-tuned to a semantic weight of 1.0. The sweep was run on the evaluation set, and selecting a parameter from it and then reporting performance at that parameter would be test-set contamination. The finding stands as evidence about a widely used default configuration, and I would rather report it honestly than harvest a point of recall from it.

What I take from this is that a component can be present, executing, consuming compute, and producing plausible logs while contributing nothing. The only reason I found it is that I ran an ablation instead of trusting the architecture diagram.

### 10.3 Multi-entity queries, and a metric that hid the problem

Comparative queries of the form "how do the contraindications of drug X and drug Y differ" fail in a specific way. In six of seven such queries, single-pass top-five retrieval returned chunks for only one of the two named drugs. The second drug is in the corpus and is retrievable. It simply loses the ranking competition, because a single query embedding cannot express a requirement for coverage across two entities.

The measurement finding is worse than the retrieval finding. My alias-based Recall@5, which counts a query as successful if any expected drug's chunk is retrieved, scored these queries at 1.000. By that metric the system was performing perfectly on the category where it was performing worst.

Entity coverage is not any-match recall, and a benchmark that does not distinguish them will report comparative-query retrieval as a solved problem.

The system nonetheless abstained on most of these queries, because confidence was low. The Refusal Guard caught, through low confidence, a retrieval failure that the retrieval metric had scored as a success. That is a genuine argument for the layered design, and an argument for reporting refusal behaviour alongside retrieval metrics rather than treating retrieval numbers as ground truth.

The remedy is query decomposition: split a multi-entity query into per-entity retrievals and merge. Documented as future work, not implemented.

---

## 11. Engineering Reflections

Some observations that do not belong in a results section but which I think are the real learning outcomes of the project.

**Building the system was the easy half.** Phases 1 to 4 produced a working, demonstrable pipeline. Phase 5, establishing whether that pipeline could be trusted, took longer than all of them together. I would now budget evaluation as the majority of the work rather than as a final step, and I would build the benchmark before building the ablations rather than alongside them.

**Ablations find bugs, not just contributions.** I built the five-configuration ablation to quantify what each component was worth. It instead found that one component was worth nothing (Section 10.2) and that another component's headline metric was measuring the wrong thing (Section 8.3). Neither would have surfaced from reading outputs, because the outputs looked fine.

**Manual annotation is a data-quality control, not only an evaluation cost.** The corpus corruption in Section 6 was found by a human reading retrieved text during annotation. Automating the annotation would have saved weeks and propagated the corruption directly into the evaluation.

**A metric that looks broken is sometimes correct about the wrong question.** When groundedness came back at 0.41, my first instinct was that the validator was miscalibrated and the threshold needed lowering. Tuning the threshold would have produced a number closer to expectation while leaving the actual defect, the denominator, untouched. I now try to reconcile a surprising metric against a hand-inspected sample before adjusting anything.

**Design assumptions get encoded as constants and then stop being visible.** The 0.55 weight on groundedness in the confidence formula encodes an assumption that evidence quality is the dominant safety signal. That assumption is stated nowhere in the code. It is a number in a formula, and the adversarial failure in Section 10.1 is that assumption failing. I would now document the reasoning behind such constants next to them.

**What I would do differently.** Build the benchmark first, including the refusal cases, and let it drive the architecture rather than the reverse. Pin document identifiers from the outset rather than resolving by name. Instrument the retrieval fusion with a diagnostic that reports how many final candidates each retriever uniquely contributed, which would have caught Section 10.2 in Phase 3 rather than Phase 5.

---

## 12. Limitations

**Annotation.** Gold chunks were annotated by a single primary annotator, with agreement assessed on a twenty-query subset only. Fleiss kappa on best-chunk selection is 0.076, and non-domain raters performed near chance on that task.

**Scale.** One therapeutic area, 28 drugs, 723 chunks, 85 queries, and only 5 adversarial items. No claim about generalisation to other therapeutic areas is supported by this data. Refusal-subset comparisons are underpowered.

**Metric independence.** Citation precision uses a 200-character snippet and is a lower bound. Its scorer shares an embedding model with the Evidence Validator, so the two are not fully independent.

**Similarity is not entailment.** The validator uses embedding cosine similarity at a fixed threshold. Similarity is insensitive to negation and reversed polarity, which are precisely the constructions that carry meaning in contraindication and warning text. A sentence asserting a drug is safe in pregnancy and a sentence asserting it is contraindicated in pregnancy are highly similar under this measure. An entailment-based validator is the correct instrument.

**Corpus gaps.** The Aubagio boxed-warning section parsed to zero chunks. One query, SP-005 on siponimod geriatric use, has no corresponding corpus content. Both are recorded and were not excluded from scoring.

**Unresolved failures.** Adversarial refusal at three of five. Multi-entity retrieval documented rather than fixed.

**No external benchmark comparison.** MIRAGE is infeasible at this compute scale and format-mismatched for abstention. I compare configurations of my own system and report BM25 as a reference floor. No claim of superiority over any external system is made.

**Monitoring without a drift study.** The dashboard exists and reads live audit logs. No drift experiment was run and no drift-detection claim is made.

---

## 13. Future Work

Ordered by how directly each follows from a measured failure.

**Safety-intent classification.** A query-side classifier operating before or alongside the evidence pipeline, addressing the adversarial gap in Section 10.1. This is the highest-value item because it targets the only failure the current architecture cannot address by tuning.

**Query decomposition for multi-entity retrieval.** Splitting comparative queries into per-entity retrievals, addressing Section 10.3. Requires a corresponding entity-coverage metric, since any-match recall will continue to report success.

**Entailment-based validation.** Replacing embedding similarity with a natural-language-inference model, addressing the negation-insensitivity limitation.

**Domain-adapted reranking.** A fine-tuned reranker motivated specifically by dosing, the weakest retrieval category at 0.50, where structured numeric content is served poorly by a general passage reranker.

**Citation-aware generation.** Constrained decoding or a post-generation realignment pass to eliminate the 23 misalignment failures at source rather than measuring them.

**Second therapeutic area.** The most direct test of whether these findings generalise or are artefacts of MS labels.

**Drift study.** The audit infrastructure supports it. It has not been run.

---

## 14. Conclusion

I set out to build a retrieval system for drug-safety questions and to demonstrate that it could be trusted. The system was the straightforward part. Demonstrating trustworthiness turned out to require building instruments that did not exist, and those instruments produced a result that inverted the assumption the project started from.

Groundedness, the metric this field treats as the proxy for trustworthiness, does not track safety. The ungoverned configuration in this project is as grounded as the governed one and answers every question it should have refused. The property that distinguishes a system I would be willing to deploy from one I would not is the presence of a decision layer that can decline, together with the evaluation apparatus that can tell whether it declines correctly. Neither is measured by any benchmark I found.

The project also produced three failures I did not anticipate and which I regard as the most valuable part of it: a corpus corruption invisible to every automated check and caught by a human reading text; a retrieval component that was mathematically incapable of contributing while appearing in every diagram and log; and a metric that was understating the system's own performance by 33 points because of how it treated the behaviour I most wanted to encourage. Each was found by measurement rather than inspection, and each would have gone unnoticed in a project that stopped at a working demonstration.

---

## 15. References

> Entries marked ⚠ require verification before final submission.

1. Amugongo, L. M., et al. Systematic review of retrieval-augmented generation in healthcare. ⚠ **Full citation required.**
2. Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. (2024). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. *International Conference on Learning Representations*, 9112 to 9141.
3. Es, S., et al. (2024). RAGAS: Automated evaluation of retrieval augmented generation. ⚠ **Confirm authors and venue.**
4. Jin, D., Pan, E., Oufattole, N., Weng, W.-H., Fang, H., and Szolovits, P. (2021). What disease does this patient have? A large-scale open domain question answering dataset from medical exams. *Applied Sciences*, 11(14), 6421.
5. Jin, Q., Dhingra, B., Liu, Z., Cohen, W., and Lu, X. (2019). PubMedQA: A dataset for biomedical research question answering. *EMNLP-IJCNLP*, 2567 to 2577.
6. Koppula, M., Madhulika, F., Sreeramoju, N., and Kolimi, P. (2025). AI-powered chatbot for FDA drug labeling information retrieval: OpenAI GPT for grounded question answering. *Analytics*, 4(4), 33.
7. Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459 to 9474.
8. Min, S., Krishna, K., Lyu, X., et al. (2023). FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. *EMNLP*, 12076 to 12100.
9. Nishisako, S., Higashi, T., and Wakao, F. (2025). Reducing hallucinations and trade-offs in responses in generative AI chatbots for cancer information: Development and evaluation study. *JMIR Cancer*, 11, e70176.
10. Pal, A., Umapathi, L. K., and Sankarasubbu, M. (2022). MedMCQA: A large-scale multi-subject multi-choice dataset for medical domain question answering. *Conference on Health, Inference, and Learning*, 248 to 260.
11. Wang, Q., Li, B., Liang, J., Shi, D., Zhang, B., and Song, Q. (2026). DrugClaw and DrugAudit: A primary-source-grounded agent and authority-aware benchmark for drug-information question answering. *arXiv:2606.01434*.
12. Xiong, G., Jin, Q., Lu, Z., and Zhang, A. (2024). Benchmarking retrieval-augmented generation for medicine. *Findings of the Association for Computational Linguistics: ACL 2024*, 6233 to 6251.
13. Yan, S.-Q., et al. (2024). Corrective retrieval augmented generation. ⚠ **Confirm authors and venue.**
14. Self-MedRAG. ⚠ **Full citation required.**
15. KnowGuard. ⚠ **Full citation required.**

---

# Appendix A: Repository Structure and Reproduction

Adapt the tree below to your actual layout before submission.

```
pharmarag/
├── ingestion/
│   ├── dailymed_client.py       # SetID-pinned SPL retrieval
│   ├── spl_parser.py            # LOINC section parsing
│   ├── chunker.py               # section-aware, 500/50
│   └── setids.yaml              # pinned document identifiers
├── retrieval/
│   ├── dense.py                 # PubMedBERT + ChromaDB
│   ├── lexical.py               # BM25
│   ├── fusion.py                # weighted RRF
│   └── rerank.py                # cross-encoder
├── agents/
│   ├── router.py                # query -> LOINC section
│   ├── validator.py             # per-sentence groundedness
│   └── guard.py                 # three-tier decision
├── generation/
│   ├── prompt.py
│   └── ollama_client.py
├── evaluation/
│   ├── benchmark/queries.jsonl  # 85 queries + gold
│   ├── metrics.py
│   ├── run_ablation.py
│   └── stats.py                 # McNemar
├── monitoring/
│   ├── dashboard.py             # Streamlit
│   └── logs/audit.jsonl
├── demos/                       # phase demo scripts, preserved
└── README.md
```

**Reproduction:**

```bash
ollama pull gemma3:12b
pip install -r requirements.txt
python -m ingestion.build_corpus --setids ingestion/setids.yaml
python -m evaluation.run_ablation --configs N,baseline,A,B,C --temp 0
python -m evaluation.stats --test mcnemar
```

---

# Appendix B: Code Blocks for Outstanding Numbers

These produce the figures marked `[RUN: block n]`. Adapt paths to your layout.

### Block 1: Corpus composition (Sections 4.1 and 6.1)

```python
import chromadb, pandas as pd
from collections import Counter

client = chromadb.PersistentClient(path="./chroma_db")   # adapt
col = client.get_collection("pharmarag_spl")             # adapt
meta = col.get(include=["metadatas"])["metadatas"]

df = pd.DataFrame(meta)
print("Total chunks:", len(df))
print("Distinct drugs:", df["drug_name"].nunique())

per_drug = (df.groupby("drug_name")
              .size().reset_index(name="chunks")
              .sort_values("chunks", ascending=False))
print(per_drug.to_markdown(index=False))

per_section = (df.groupby("section_name")
                 .size().reset_index(name="chunks")
                 .sort_values("chunks", ascending=False))
print(per_section.to_markdown(index=False))

print("\nDrugs with zero chunks in a core section (corpus gaps):")
core = ["boxed_warning","contraindications","warnings_and_precautions",
        "dosage_and_administration","adverse_reactions"]
for d in df["drug_name"].unique():
    have = set(df[df["drug_name"]==d]["section_name"])
    missing = [s for s in core if s not in have]
    if missing:
        print(" ", d, "->", missing)
```

Paste `per_drug` and `per_section` as tables into Section 6.1. The gap listing confirms the Aubagio boxed-warning finding in Section 12.

### Block 2: Per-category query counts (Table 7.1)

```python
import json, pandas as pd

rows = [json.loads(l) for l in open("evaluation/benchmark/queries.jsonl")]
df = pd.DataFrame(rows)

counts = (df.groupby(["category","expected_decision"])
            .size().reset_index(name="n")
            .sort_values(["expected_decision","n"], ascending=[True,False]))
print(counts.to_markdown(index=False))
print("\nTotal:", len(df))
print("Answerable:", (df.expected_decision!="refuse").sum())
print("Refuse:", (df.expected_decision=="refuse").sum())
print("With gold chunk:", df.gold_chunk.notna().sum())
```

### Block 3: Per-category Recall@5 (Figure 2, and the dosing 0.50 claim)

```python
import json, pandas as pd

res = [json.loads(l) for l in open("evaluation/results/config_C.jsonl")]  # adapt
df = pd.DataFrame(res)
ans = df[df.expected_decision != "refuse"]

per_cat = (ans.groupby("category")
              .agg(n=("query_id","count"),
                   recall_at_5=("recall_at_5","mean"),
                   gold_at_5=("gold_chunk_at_5","mean"))
              .round(3).sort_values("recall_at_5"))
print(per_cat.to_markdown())
```

Confirms which category is weakest. My handoff records dosing at 0.50; verify against this output before citing it.

### Block 4: Latency distribution including P95 (Table 9.6)

```python
import json, numpy as np, pandas as pd

logs = [json.loads(l) for l in open("monitoring/logs/audit.jsonl")]
df = pd.DataFrame([r["latency"] for r in logs])
df["end_to_end"] = df.sum(axis=1)

summary = pd.DataFrame({
    "mean_s": df.mean().round(2),
    "median_s": df.median().round(2),
    "p95_s": df.quantile(0.95).round(2),
    "max_s": df.max().round(2),
}).sort_values("mean_s", ascending=False)
print(summary.to_markdown())
```

Replaces the `[RUN: block 4]` marker in Section 3.3 and completes Table 9.6.

### Block 5: Confidence distribution by decision tier (optional Figure 6)

```python
import json, pandas as pd, matplotlib.pyplot as plt

logs = [json.loads(l) for l in open("monitoring/logs/audit.jsonl")]
df = pd.DataFrame([{"confidence": r["confidence"], "decision": r["decision"]}
                   for r in logs])

fig, ax = plt.subplots(figsize=(8,4.5))
for dec in ["INSUFFICIENT_EVIDENCE","ANSWER_WITH_CAUTION","ANSWER"]:
    sub = df[df.decision==dec]["confidence"]
    ax.hist(sub, bins=25, alpha=0.65, label=f"{dec} (n={len(sub)})")
ax.axvline(0.45, ls="--", c="0.3"); ax.axvline(0.65, ls="--", c="0.3")
ax.set_xlabel("Confidence C"); ax.set_ylabel("Queries")
ax.legend(); fig.tight_layout()
fig.savefig("figures/fig6_confidence.png", dpi=200)
```

Useful for showing the thresholds are separating cases rather than cutting through a single mass.

---

# Appendix C: Figure and Screenshot Inventory

## Figures to generate

**FIGURE 1: System architecture.** Section 4.
Tool: draw.io or Excalidraw, exported to PNG at 300 dpi.
Content: left to right, User query → Query Router (box labelled with LOINC section output) → parallel branch to BM25 and Dense/ChromaDB → RRF fusion (annotate "k=60, 0.6/0.4") → Cross-encoder rerank (annotate "top-20 → top-5") → Gemma 3 12B via Ollama (annotate "local, temp 0") → Evidence Validator (annotate "per-sentence, τ=0.35") → Refusal Guard (three arrows out: ANSWER, ANSWER_WITH_CAUTION, INSUFFICIENT_EVIDENCE) → Response. A dashed line from every stage down to an "Audit log (JSONL)" box. Shade the three agent boxes in one colour so the governance layer is visually distinct from the retrieval pipeline.

**FIGURE 2: Per-category Recall@5 across configurations.** Section 9.1.
Data: Block 3, run per configuration.
Type: grouped horizontal bar chart, categories on the y-axis sorted ascending by Config C recall, one bar per configuration. Annotate the dosing bar. Sorting ascending puts the weakest category at the top where it reads first.

**FIGURE 3: Groundedness against unsafe emission.** Section 9.2. This is the most important figure in the report.
Data: Table 9.2.
Type: two panels side by side sharing an x-axis of configuration (N, Baseline, C). Left panel plots groundedness on a y-axis fixed to 0 to 1, producing a visibly flat line. Right panel plots unsafe emission rate on the same 0 to 1 axis, producing a cliff. Keeping both axes at 0 to 1 is the whole point; do not autoscale, because autoscaling the left panel will manufacture a slope that is not there.

**FIGURE 4: Semantic weight sweep.** Section 10.2.
Data: Table 10.1.
Type: line plot, semantic weight on x from 0.4 to 1.0, Recall@5 on y. Draw a horizontal dashed line at 0.776 labelled "dense-only". Shade the region from 0.9 to 1.0 to show the plateau. Mark the deployed setting at 0.6 with an annotation reading "deployed configuration".

**FIGURE 5: Latency breakdown.** Section 9.5.
Data: Table 9.6 and Block 4.
Type: horizontal stacked bar, one bar per configuration (Baseline and C), segments for routing, retrieval, generation, validation. Makes the point that generation dominates and reranking is affordable.

**FIGURE 6 (optional): Confidence distribution by decision.** Block 5.

## Screenshots to capture

**SCREENSHOT 1: Local inference.** Section 4.5.
Terminal showing `ollama list` with gemma3:12b present, and `ollama ps` with the model loaded. Supports the local-only governance claim.

**SCREENSHOT 2: Audit record.** Section 4.7.
One JSONL record pretty-printed. Redact nothing; the point is completeness. Ensure the visible fields include request_id, routed_section, retrieved chunk ids with scores, per-stage latency, groundedness, confidence, decision, and reason.

**SCREENSHOT 3: A well-supported answer.** Section 5, Phase 4.
Suggested query: "What are the contraindications for natalizumab?" Capture the full response with numbered citations, the evidence table, and a visible ANSWER decision with its confidence score.

**SCREENSHOT 4: A correct refusal.** Section 5, Phase 4.
Suggested query: a near-scope one, such as "What is the recommended dose of metformin?" Capture the INSUFFICIENT_EVIDENCE decision with the confidence score and the stated reason. Include the confidence value so the reader can see it falls below 0.45.

**SCREENSHOT 5: Answer with caution.** Section 5, Phase 4.
Any query landing between 0.45 and 0.65. Worth hunting for a genuine one rather than forcing it, since the middle tier is the hardest to demonstrate.

**SCREENSHOT 6: Monitoring dashboard.** Section 5, Phase 5.
Full dashboard view showing decision-tier distribution, latency over time, and the recent-requests table. Populate with at least a few dozen requests first so the charts are not sparse.

**SCREENSHOT 7 (strongly recommended, add to Section 10.1): The adversarial failure.**
Capture the system *answering* one of the two adversarial queries it fails on, with the high groundedness score and the ANSWER decision both visible. This single image makes the central argument of the report better than any prose can: an answer that is well grounded, correctly cited, and unsafe. Place it directly beneath the Section 10.1 heading.

**SCREENSHOT 8 (recommended, add to Section 6.2): The corpus failure.**
If you still have the pre-fix index or can reconstruct it, capture a retrieval result showing the Kesimpta query returning ARZERRA oncology dosing content. If not reconstructible, show the `setids.yaml` fix instead with a comment marking the three corrected entries.

---

# Appendix D: Full Query Benchmark

Include the complete 85-query benchmark as a table: query id, category, expected decision, query text, and gold chunk identifier. Generate with:

```python
import json, pandas as pd
rows = [json.loads(l) for l in open("evaluation/benchmark/queries.jsonl")]
df = pd.DataFrame(rows)[["query_id","category","expected_decision",
                         "query_text","gold_chunk_id"]]
print(df.to_markdown(index=False))
```

For an appendix this long, consider including only the 27 refuse queries in full, since those are the novel part of the artefact, and providing the remainder as a supplementary file.
