import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

client_faq = Groq()
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

_FAQ_ROWS: List[Dict[str, str]] = []
_FAQ_TEXTS: List[str] = []
_FAQ_INDEX = None
_FAQ_BM25 = None

faq_prompt = """You are a helpful ecommerce support assistant.
Answer the user's question using only the FAQ contexts provided.
Keep the answer short, direct, and natural.
If the contexts do not answer the question, say that you could not find a matching FAQ.
"""


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _expand_query(question: str) -> str:
    q = question.lower()
    expansions = []
    synonym_groups = {
        "parcel package delivery shipment tracking order status track shipped": [
            "parcel",
            "package",
            "shipment",
            "shipping status",
            "where is my order",
        ],
        "damaged broken defective faulty replacement refund": [
            "broken",
            "defective",
            "faulty",
        ],
        "payment pay upi card cash cod net banking checkout": [
            "pay",
            "payment",
            "upi",
            "card",
            "cash",
            "cod",
        ],
        "promo code coupon discount offer deals": [
            "coupon",
            "promo",
            "discount code",
        ],
        "international shipping outside india abroad": [
            "outside india",
            "abroad",
            "international delivery",
        ],
    }

    for expansion, triggers in synonym_groups.items():
        if any(trigger in q for trigger in triggers):
            expansions.append(expansion)

    if not expansions:
        return question
    return f"{question} {' '.join(expansions)}"


def _embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL)


def ingest_faq_data(csv_path: Path) -> None:
    """Load FAQ CSV rows and build dense + sparse indexes for hybrid retrieval."""
    global _FAQ_ROWS, _FAQ_TEXTS, _FAQ_INDEX, _FAQ_BM25

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"FAQ file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"question", "answer"}
    if not required_columns.issubset(df.columns):
        raise ValueError("FAQ CSV must contain 'question' and 'answer' columns")

    normalized_df = (
        df[["question", "answer"]]
        .dropna()
        .astype(str)
        .apply(lambda column: column.str.strip())
    )

    _FAQ_ROWS = [
        {"id": str(idx), "question": row.question, "answer": row.answer}
        for idx, row in enumerate(normalized_df.itertuples(index=False))
        if row.question and row.answer
    ]

    if not _FAQ_ROWS:
        _FAQ_TEXTS = []
        _FAQ_INDEX = None
        _FAQ_BM25 = None
        return

    _FAQ_TEXTS = [
        f"Question: {row['question']}\nAnswer: {row['answer']}"
        for row in _FAQ_ROWS
    ]

    embeddings = _embed_texts(_FAQ_TEXTS)
    _FAQ_INDEX = faiss.IndexFlatIP(embeddings.shape[1])
    _FAQ_INDEX.add(embeddings)

    _FAQ_BM25 = BM25Okapi([_tokenize(text) for text in _FAQ_TEXTS])


def _rrf_fuse(rankings: List[List[int]], k: int = 60) -> Dict[int, float]:
    fused_scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
    return fused_scores


def _dense_rank(question: str, pool_size: int) -> Dict[int, float]:
    if _FAQ_INDEX is None:
        return {}

    query_embedding = _embed_texts([question])
    scores, indices = _FAQ_INDEX.search(query_embedding, pool_size)

    return {
        int(index): float(score)
        for score, index in zip(scores[0], indices[0])
        if index >= 0
    }


def _sparse_rank(question: str, pool_size: int) -> Dict[int, float]:
    if _FAQ_BM25 is None:
        return {}

    scores = _FAQ_BM25.get_scores(_tokenize(question))
    ranked_indices = np.argsort(scores)[::-1][:pool_size]

    return {
        int(index): float(scores[index])
        for index in ranked_indices
        if scores[index] > 0
    }


def retrieve_faq(question: str, top_k: int = 3, pool_size: int = 8):
    """Hybrid FAQ retrieval: FAISS dense search + BM25 + RRF + cross-encoder rerank."""
    if not _FAQ_ROWS:
        return []

    expanded_question = _expand_query(question)
    pool_size = min(pool_size, len(_FAQ_ROWS))
    dense_scores = _dense_rank(expanded_question, pool_size)
    sparse_scores = _sparse_rank(expanded_question, pool_size)

    dense_ranking = sorted(dense_scores, key=dense_scores.get, reverse=True)
    sparse_ranking = sorted(sparse_scores, key=sparse_scores.get, reverse=True)
    fused_scores = _rrf_fuse([dense_ranking, sparse_ranking])

    if not fused_scores:
        return []

    candidate_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:pool_size]

    try:
        pairs = [(expanded_question, _FAQ_TEXTS[index]) for index in candidate_ids]
        cross_scores = _get_cross_encoder().predict(pairs)
    except Exception:
        cross_scores = [0.0 for _ in candidate_ids]

    candidates = []
    for index, cross_score in zip(candidate_ids, cross_scores):
        row = _FAQ_ROWS[index]
        candidates.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "dense_score": dense_scores.get(index, 0.0),
                "bm25_score": sparse_scores.get(index, 0.0),
                "rrf_score": fused_scores.get(index, 0.0),
                "cross_score": float(cross_score),
            }
        )

    return sorted(candidates, key=lambda item: item["cross_score"], reverse=True)[:top_k]


def _generate_faq_answer(question: str, retrieved_faqs: List[Dict[str, str]]) -> str:
    contexts = "\n\n".join(
        (
            f"FAQ {idx}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )
        for idx, item in enumerate(retrieved_faqs, start=1)
    )

    chat_completion = client_faq.chat.completions.create(
        messages=[
            {"role": "system", "content": faq_prompt},
            {
                "role": "user",
                "content": f"USER QUESTION: {question}\n\nRETRIEVED FAQ CONTEXTS:\n{contexts}",
            },
        ],
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=256,
    )

    return chat_completion.choices[0].message.content


def faq_chain(question: str, min_dense_score: float = 0.25) -> str:
    matches = retrieve_faq(question, top_k=3)
    if not matches:
        return "FAQ data is not loaded yet."

    best_match = matches[0]
    if best_match["dense_score"] < min_dense_score and best_match["bm25_score"] <= 0:
        return (
            "I could not find a matching FAQ for that. "
            "Please rephrase your question or ask about returns, refunds, payments, tracking, or offers."
        )

    try:
        return _generate_faq_answer(question, matches)
    except Exception:
        return best_match["answer"]


if __name__ == "__main__":
    faqs_path = Path(__file__).parent / "resources/faq_data.csv"
    ingest_faq_data(faqs_path)
    print(faq_chain("Can I pay with UPI?"))
