import threading
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from audit import log_interaction
from generate import LLM_MODEL, NOT_FOUND, answer_question
from privacy import Redactor
from retriever import Retriever
from verify import ClaimVerifier
from fastapi.responses import RedirectResponse

RERANK = "bge-m3"
CANDIDATES = 10
CONFIG = {"llm": LLM_MODEL, "prompt": "v3", "embedder": "e5-small",
          "rerank": RERANK, "candidates": CANDIDATES}

state = {}
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    state["retriever"] = Retriever("e5-small", "hybrid", rerank_key=RERANK, candidates=CANDIDATES)
    state["verifier"] = ClaimVerifier()
    state["redactor"] = Redactor()
    yield
    state.clear()


app = FastAPI(title="Offline DocQA", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    verify: bool = True


class Source(BaseModel):
    n: int
    doc: str
    page_start: int
    page_end: int


class Claim(BaseModel):
    claim: str
    level: str
    score: float


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]
    claims: list[Claim]
    invalid_citations: list[int]
    retrieval_s: float
    generation_s: float


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "retriever" in state, "llm": LLM_MODEL}
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    with lock:
        try:
            r = answer_question(state["retriever"], req.question)
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="LLM server (Ollama) is not reachable")

        claims = []
        if req.verify and not r["refused"]:
            claims = state["verifier"].verify(r["answer"], r["results"])

        redactor = state["redactor"]
        log_interaction(redactor.redact(req.question)[0], redactor.redact(r["answer"])[0],
                        r, claims, {**CONFIG, "verify": req.verify})

    sources = []
    for n in r["cited"]:
        chunk, _ = r["results"][n - 1]
        sources.append(Source(n=n, doc=chunk["doc"],
                              page_start=chunk["page_start"], page_end=chunk["page_end"]))

    return AskResponse(
        answer=NOT_FOUND if r["refused"] else r["answer"],
        refused=r["refused"],
        sources=sources,
        claims=[Claim(claim=c["claim"], level=c["level"], score=c["score"]) for c in claims],
        invalid_citations=r["invalid_citations"],
        retrieval_s=round(r["retrieval_s"], 2),
        generation_s=round(r["generation_s"], 2),
    )