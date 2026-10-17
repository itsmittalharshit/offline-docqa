# Experiment Results — Offline Document QA

**Hardware:** MacBook Air M4 (16 GB), everything runs locally.

**Pipeline:** ingestion → cleaning → chunking → hybrid search → reranking → cited generation → claim verification.

**Corpus:** 3 RBI reports, 331 chunks in total.

| File | Report | Pages | Chunks |
|---|---|---|---|
| report1.pdf | Internal Working Group to Review the Liquidity Management Framework (July 2025) | 57 | 85 |
| report2.pdf | National Strategy for Financial Inclusion 2025-30 | 74 | 147 |
| report3.pdf | Advisory Committee on Ways and Means Advances to State Governments | 75 | 99 |

**Chunking:** sentence-packed, max 250 words, 50-word overlap. Chunks may cross pages and record `page_start` / `page_end`.

**Retrieval metrics:**
- **Recall@k:** the fraction of questions with a correct chunk in the top k. A chunk is correct if it comes from the labelled document and its page range covers a labelled PDF page.
- **MRR:** the mean of 1/rank of the first correct chunk; a miss counts as 0.

**Generation metrics:** see Phase 6.

## Eval set versions

| Version | Questions | Notes |
|---|---|---|
| v1 | 14 (q01–q14) | First set. Mostly reports 1 and 3; much of the wording is copied from the documents, so scores are likely optimistic. |
| v2 | 38 answerable + 4 unanswerable | All 3 reports. Labels use PDF page indexes, not printed page numbers, and were validated with `check_eval.py` (evidence phrases). |
| v2.1 | 38 answerable + 4 unanswerable | q28 label fixed (SNA-SPARSH). Used for Phases 5–7. |

Results from different eval versions are **not directly comparable**.

---

## Phase 1 — Baseline retrieval (eval v1, 14 questions)

| Config | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| dense, multilingual-e5-small | 0.50 | 0.79 | 0.79 | 0.86 | 0.640 |

Paraphrase and Hindi questions are the weak spots. Direct and recommendation questions work well.

## Phase 2 — Embedding model comparison (eval v1)

| Config | R@1 | R@3 | R@5 | R@10 | MRR | Embedding time, 331 chunks |
|---|---|---|---|---|---|---|
| dense, e5-small (384-d, 118M) | **0.50** | **0.79** | 0.79 | 0.86 | **0.640** | ~5 s |
| dense, bge-m3 (1024-d, 568M) | 0.43 | 0.64 | 0.79 | **0.93** | 0.561 | ~44 s |

bge-m3 is not better on top ranks and is about 10× slower. With only 14 questions, the differences are mostly noise.

## Phase 3 — Hybrid search (eval v1)

| Config | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| dense, e5-small | 0.50 | 0.79 | 0.79 | 0.86 | 0.640 |
| bm25 (SQLite FTS5) | 0.29 | 0.43 | 0.57 | 0.79 | 0.407 |
| hybrid (RRF), e5-small | 0.50 | 0.71 | 0.86 | 0.86 | 0.627 |
| dense, bge-m3 | 0.43 | 0.64 | 0.79 | 0.93 | 0.561 |
| hybrid (RRF), bge-m3 | 0.43 | 0.71 | 0.79 | 0.86 | 0.598 |

- BM25 helps on distinctive terms (q09 "2021", q10 "28 State Governments").
- It struggles where the key terms are common within the report ("WMA", "SDF").
- It fails completely on Hindi questions.

## Phase 4 — Re-run on the larger eval set (eval v2, 38 questions)

| Config | R@1 | R@3 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|---|
| dense, e5-small | **0.63** | **0.82** | 0.82 | 0.89 | **0.97** | **0.723** |
| bm25 | 0.58 | 0.74 | 0.82 | 0.92 | 0.92 | 0.676 |
| hybrid, e5-small (Hindi routing) | 0.61 | 0.79 | **0.84** | **0.95** | **0.97** | 0.708 |
| dense, bge-m3 | 0.55 | 0.76 | 0.84 | 0.89 | 0.95 | 0.667 |
| hybrid, bge-m3 | 0.55 | 0.79 | 0.84 | 0.92 | 0.97 | 0.688 |

- bge-m3 is confirmed not better than e5-small on 38 questions.
- BM25 is much stronger on report2, which has distinctive terms such as UPI LITE, ULI and "one billion".
- The Hindi routing rule (skip BM25 for mostly non-Latin queries) hurt q20: rank 16 with hybrid vs rank 2 with BM25.

## Phase 5 — BM25 Latin-token fix + reranking (eval v2.1, 38 questions)

| Config | R@1 | R@3 | R@5 | R@10 | R@20 | MRR | Latency |
|---|---|---|---|---|---|---|---|
| hybrid, e5-small (BM25 on Latin tokens only) | 0.61 | 0.79 | 0.84 | **1.00** | **1.00** | 0.714 | **14 ms** |
| + rerank mmarco-mMiniLMv2-L12, top 20 | 0.68 | 0.84 | 0.87 | 0.95 | 1.00 | 0.773 | 280 ms |
| + rerank bge-reranker-v2-m3, top 20 | **0.74** | 0.84 | 0.87 | 0.97 | 1.00 | **0.807** | 2,808 ms |
| + rerank bge-reranker-v2-m3, top 10 | — | — | — | — | — | — | ~1.5–1.9 s (observed in app) |

- Feeding BM25 only the Latin-script words fixed q20 (rank 16 → 6). Hybrid now has Recall@10 = 1.00.
- The rerankers lift Recall@1 from 0.61 to 0.74.
- Rerankers can also hurt: minilm pushed q26 from rank 7 to 14. Both rerankers pushed q16 down, which suggests incomplete labels for q16.
- Reranking only the top 10 is justified because stage-1 R@10 = 1.00. The full evaluation run for top 10 is still to be done; only latency has been observed so far.

## Phase 6 — Answer generation (eval v2.1; 38 answerable + 4 unanswerable)

**Generation setup:**
- Model: qwen2.5:7b (4-bit) via Ollama.
- `temperature` 0, `num_ctx` 8192.
- Top 5 reranked chunks as context.

**Generation metrics:**
- **False-refusal rate:** answerable questions answered with "not found".
- **Hallucination (unanswerable):** unanswerable questions answered anyway.
- **Correctness:** a heuristic — the gold answer's key numbers or words appear in the answer.
- **Cites a correct page:** at least one cited passage is from a labelled page.
- **Citation precision:** the share of cited passages that are from labelled pages.
- **Language OK:** the answer's script matches the question's.
- **Invented citations:** answers citing passage numbers that don't exist.

| Prompt | False refusal | Halluc. (unans.) | Correctness | Cites correct page | Citation precision | Language OK | Invented cites | Gen time (warm) |
|---|---|---|---|---|---|---|---|---|
| v1 *(old heuristic)* | 0.03 | 0.00 | 0.86 | 0.84 | 0.66 | — | — | — |
| v2 *(old heuristic)* | 0.03 | 0.00 | 0.84 | 0.76 | 0.62 | — | — | — |
| v1 (re-scored) | 0.03 | 0.00 | 0.89 | 0.84 | **0.66** | 0.97 | 0.08 | ~2.5 s |
| **v3** | 0.03 | 0.00 | 0.89 | **0.89** | 0.65 | **1.00** | **0.00** | ~3–4 s |

**Prompt versions:**
- **v1:** 5 short rules — use only the passages, cite, refuse with a fixed sentence, same language, concise.
- **v2:** v1 plus stricter citation rules and "distinguish existing vs recommended values". **Made things worse:** the model copied passages, put citations first and invented citation numbers.
- **v3:** v1 rules plus one instruction placed **after** the question: an explicit language ("Answer in Hindi/English"), citations at the end of sentences, and the number of passages that exist. This fixed the Chinese reply to the Hindi q20 and removed invented citations in evaluation.

**Findings:**
- **The heuristic correctness metric is unreliable in both directions.**
  - False passes: q13 invented "Securities Finance Companies" for SPD; q20 was answered in Chinese under v1.
  - False fails: q14 has the name in Devanagari; q37 differs only in comma format.
  - After adding the number normalization and language checks, about 5 genuinely wrong answers remain: q01, q03, q04, q13 and q14.
- **These errors survive every prompt version**, so they are not prompt problems; they need claim verification.
- **Timing:** the first evaluation run after loading the models was always slow (~13–15 s per question) regardless of prompt. This was a warm-up / memory-pressure effect. When warm, generation takes about 2.5–4 s per answer at ~21 tokens/s, and most of that time is reading the ~2k-token context (prefill). Fixed with an untimed warm-up call.
- **Temperature 0 is not fully deterministic on a GPU.** An invented citation ([6], with only 5 passages) reappeared once in the interactive app after 0.00 in evaluation, so the runtime citation check stays.

## Phase 7 — Claim verification (NLI)

**Method:**
- Each answer sentence is one claim.
- Each claim is checked against its cited passages with `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`. The score is the entailment probability.
- Uncited sentences are checked against all passages cited in the answer.
- An answer's score is its weakest claim.

**Ground truth for evaluating the verifier:** 5 known-wrong answers (q01, q03, q04, q13, q14) vs 32 correct answers, judged by reading the answers.

| Verifier version | Wrong caught (of 5) | False alarms (of 32) | Answers flagged |
|---|---|---|---|
| v1: whole chunk as premise, threshold 0.5 | **5** | 10 | 41% |
| v2: 2-sentence windows, number normalization, abbreviation-safe splitting, threshold 0.5 | 2 | **2** | 11% |

**Verifier v1 failure causes:**
- Long premises dilute the entailment signal.
- Sentences were split at "Dr." and "डॉ.", leaving fragments.
- Number formats differed ("1,000" vs "1000").

**Threshold sweep, verifier v2** (lowest claim score per answer):

| Threshold | Wrong caught | False alarms |
|---|---|---|
| 0.30 | 2/5 | 1 |
| 0.50 | 2/5 | 2 |
| 0.70 | 3/5 | 3 |
| 0.80 | 4/5 | 5 |
| 0.90 | 4/5 | 9 |
| 0.95 | 5/5 | 9 |

- **AUROC ≈ 0.90:** a wrong answer scores below a correct one in 144 of 160 pairs.
- **No single threshold separates wrong from correct answers.** Confident wrong answers like q03 (0.947, which mixes up two similar rules) score above several correct ones.

---

## Decisions

1. **Embedding model: multilingual-e5-small.**
   - bge-m3 was never better on top ranks, and it is about 10× slower with 2.7× larger vectors.
   - The bge-m3 embedding model was removed to save disk space.
2. **First stage: hybrid (dense + BM25, fused with RRF, 50 candidates from each).**
   - It has the best recall at depth: R@10 = 1.00.
3. **BM25 is fed Latin-script tokens only**, instead of a routing rule.
   - Hindi questions that contain English terms still benefit from BM25.
   - Pure Hindi queries fall back to dense ranking automatically.
4. **Reranker: bge-reranker-v2-m3 on the top 10 candidates.**
   - It gives the best quality. minilm is kept as a "fast mode" (`--rerank minilm`).
5. **LLM: qwen2.5:7b with prompt v3.**
   - It has the best citation and language behaviour at no extra cost.
   - Prompts are versioned in code (`SYSTEM_PROMPTS`), never overwritten.
6. **Claim verification uses three confidence levels instead of one cutoff.**

   | Level | Score | Meaning shown to the user |
   |---|---|---|
   | ❌ unsupported | < 0.5 | Not supported by the cited sources |
   | ⚠️ weak | 0.5–0.9 | Partially supported — check the source |
   | ✅ verified | ≥ 0.9 | Supported by the cited sources |

   A false alarm at ⚠️ only asks the user to check the source, which is cheap. Only ❌ makes a strong claim.

## Known limitations and open issues

- **Confident wrong answers pass verification.**
  - q03 scores 0.947. NLI checks *faithfulness to the passage*, not whether the answer addresses the question or mixes up similar entities.
  - q01 is caught only because its wording ("revised aggregate limit") does not match the passage.
- **Verifier false alarms**, e.g. q26 at 0.23 for a correct "one billion users by December 2029" answer. Check the passage wording (`find.py report2.pdf "billion"`).
- **Thresholds were chosen on the same 38 questions they are evaluated on**, with only 5 known-wrong answers, which is mild overfitting. Future work: write more questions and keep a held-out test set.
- **q30** is the only false refusal (IWG report submission date). It is not yet diagnosed whether this is a retrieval or a generation problem.
- **q01 is an ambiguous question.** "Revised" can mean the current limit or the recommended one. Consider rewording it to "recommended aggregate WMA limit" in the next eval version.
- **q16:** the labels are probably incomplete; the reranker consistently promotes other pages.
- **Hindi output quality:** occasional misspellings (q20 "वजिरित") and wrong phrasing (q14 "named after" instead of "chaired by").
- **Correctness is still a heuristic.** Reading answers remains necessary; an LLM-as-judge could be explored later.
- **The eval set is small.** One question equals 2.6% of a metric.
