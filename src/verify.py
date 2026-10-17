import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
CITATION = re.compile(r"\[(\d+)\]")
CITES_AFTER_STOP = re.compile(r"([.!?।])\s*((?:\[\d+\])+)")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")
ABBREVIATION = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Shri|Smt|Prof|No|viz|etc|vs)\.|\b(?:i\.e|e\.g)\.|डॉ\.",
    re.IGNORECASE,
)
THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d)")
PLACEHOLDER = "§"


def split_sentences(text):
    """Split into sentences without breaking on abbreviations like 'Dr.'."""
    protected = ABBREVIATION.sub(lambda m: m.group(0).replace(".", PLACEHOLDER), text)
    parts = SENTENCE_SPLIT.split(protected)
    return [p.replace(PLACEHOLDER, ".").strip() for p in parts if p.strip()]


def windows(text, size=2):
    """Overlapping windows of `size` sentences."""
    sents = split_sentences(text)
    if len(sents) <= size:
        return [" ".join(sents)]
    return [" ".join(sents[i:i + size]) for i in range(len(sents) - size + 1)]


def normalize_numbers(text):
    return THOUSANDS_COMMA.sub("", text)


class ClaimVerifier:
    def __init__(self, device="mps", low=0.5, high=0.9):
        self.device = device
        self.low = low
        self.high = high
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.model.to(device).eval()
        labels = {name.lower(): idx for idx, name in self.model.config.id2label.items()}
        self.entail_idx = labels["entailment"]

    def level(self, score):
        if score >= self.high:
            return "verified"
        if score >= self.low:
            return "weak"
        return "unsupported"

    def best_entailment(self, premises, hypothesis):
        """Highest entailment probability of `hypothesis` over many premises (one batch)."""
        inputs = self.tokenizer(premises, [hypothesis] * len(premises),
                                truncation="only_first", max_length=512,
                                padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)
        return probs[:, self.entail_idx].max().item()

    def verify(self, answer, results):
        text = CITES_AFTER_STOP.sub(r"\2\1", answer.strip())   # "x. [1]" -> "x [1]."
        all_cited = {int(n) for n in CITATION.findall(text) if 1 <= int(n) <= len(results)}

        claims = []
        for sentence in split_sentences(text):
            claim = CITATION.sub("", sentence)
            claim = re.sub(r"\s+([.।])", r"\1", claim).strip()   # "x ." -> "x."
            if len(claim.split()) < 3:
                continue

            cited = {int(n) for n in CITATION.findall(sentence) if 1 <= int(n) <= len(results)}
            sources = cited or all_cited
            if not sources:
                claims.append({"claim": claim, "score": 0.0,
                               "supported": False, "level": "unsupported"})
                continue

            premises = [w for n in sorted(sources)
                        for w in windows(normalize_numbers(results[n - 1][0]["text"]))]
            score = self.best_entailment(premises, normalize_numbers(claim))
            claims.append({"claim": claim, "score": round(score, 3),
                           "supported": score >= self.low, "level": self.level(score)})
        return claims