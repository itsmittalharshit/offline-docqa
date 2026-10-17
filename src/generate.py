import re
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen2.5:7b"
CITATION = re.compile(r"\[(\d+)\]")
NOT_FOUND = "I could not find this in the documents."

RULES_V1 = f"""You answer questions about internal reports.
Rules:
1. Use ONLY the numbered context passages. Do not use outside knowledge.
2. After each sentence that states a fact, cite its passage number(s), like [1] or [2][3].
3. If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"
4. Answer in the same language as the question.
5. Be concise: at most 4 sentences."""

RULES_V2 = f"""You answer questions about internal reports.
Rules:
1. Use ONLY the numbered context passages. Do not use outside knowledge.
2. Cite only the passage(s) that directly state each fact, like [1] or [2][3]. Never cite a passage that does not contain the fact.
3. If the passages give several values for the same thing (for example an existing limit and a recommended or revised limit), state clearly which value is which.
4. If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"
5. Answer in the same language as the question.
6. Be concise: at most 4 sentences."""

SYSTEM_PROMPTS = {"v1": RULES_V1, "v2": RULES_V2, "v3": RULES_V1}

LANGUAGE_NAMES = {"devanagari": "Hindi (हिंदी में उत्तर दें)", "latin": "English"}


def main_script(text):
    """Which writing system most letters use: 'devanagari', 'latin' or 'other'."""
    counts = {"devanagari": 0, "latin": 0, "other": 0}
    for c in text:
        if not c.isalpha():
            continue
        if "\u0900" <= c <= "\u097f":
            counts["devanagari"] += 1
        elif c.isascii():
            counts["latin"] += 1
        else:
            counts["other"] += 1
    return max(counts, key=counts.get)


def build_context(results):
    blocks = []
    for i, (chunk, _score) in enumerate(results, start=1):
        if chunk["page_start"] == chunk["page_end"]:
            pages = f"p.{chunk['page_start']}"
        else:
            pages = f"pp.{chunk['page_start']}-{chunk['page_end']}"
        blocks.append(f"[{i}] ({chunk['doc']}, {pages})\n{chunk['text']}")
    return "\n\n".join(blocks)


def user_message(question, context, n_passages, prompt_version):
    msg = f"Context:\n{context}\n\nQuestion: {question}"
    if prompt_version == "v3":
        lang = LANGUAGE_NAMES.get(main_script(question), "the same language as the question")
        msg += (f"\n\nAnswer in {lang}. Put citations like [1] at the END of each sentence. "
                f"Only passages [1] to [{n_passages}] exist.")
    return msg


def ask_llm(question, context, n_passages, model=LLM_MODEL, prompt_version="v1"):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
            {"role": "user", "content": user_message(question, context, n_passages, prompt_version)},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    tokens = data.get("eval_count", 0)
    seconds = data.get("eval_duration", 1) / 1e9
    return data["message"]["content"], tokens / seconds


def check_citations(answer, n_passages):
    numbers = sorted({int(n) for n in CITATION.findall(answer)})
    valid = [n for n in numbers if 1 <= n <= n_passages]
    invalid = [n for n in numbers if n not in valid]
    return valid, invalid


def answer_question(retriever, question, k=5, prompt_version="v1"):
    t0 = time.perf_counter()
    results = retriever.search(question, k=k)
    t1 = time.perf_counter()

    answer, tokens_per_sec = ask_llm(question, build_context(results), len(results),
                                     prompt_version=prompt_version)
    t2 = time.perf_counter()

    cited, invalid = check_citations(answer, len(results))
    return {
        "answer": answer,
        "results": results,
        "cited": cited,
        "invalid_citations": invalid,
        "refused": NOT_FOUND.lower() in answer.lower(),
        "retrieval_s": t1 - t0,
        "generation_s": t2 - t1,
        "tokens_per_s": tokens_per_sec,
    }