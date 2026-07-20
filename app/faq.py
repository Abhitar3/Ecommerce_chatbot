import os
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

client_faq = Groq()
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

_FAQ_ROWS: List[Dict[str, str]] = []
_FAQ_INDEX = None

faq_prompt = """You are a helpful ecommerce support assistant.
Answer the user's question using only the FAQ context provided.
Keep the answer short, direct, and natural.
If the context does not answer the question, say that you could not find a matching FAQ.
"""


def _embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def ingest_faq_data(csv_path: Path) -> None:
    """Load FAQ CSV rows and build an in-memory FAISS vector index."""
    global _FAQ_ROWS, _FAQ_INDEX

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
        {"question": row.question, "answer": row.answer}
        for row in normalized_df.itertuples(index=False)
        if row.question and row.answer
    ]

    if not _FAQ_ROWS:
        _FAQ_INDEX = None
        return

    faq_texts = [
        f"Question: {row['question']}\nAnswer: {row['answer']}"
        for row in _FAQ_ROWS
    ]
    embeddings = _embed_texts(faq_texts)

    _FAQ_INDEX = faiss.IndexFlatIP(embeddings.shape[1])
    _FAQ_INDEX.add(embeddings)


def _retrieve_faq(question: str, top_k: int = 1):
    if _FAQ_INDEX is None or not _FAQ_ROWS:
        return []

    query_embedding = _embed_texts([question])
    scores, indices = _FAQ_INDEX.search(query_embedding, top_k)

    matches = []
    for score, index in zip(scores[0], indices[0]):
        if index < 0:
            continue
        row = _FAQ_ROWS[index]
        matches.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "score": float(score),
            }
        )
    return matches


def _generate_faq_answer(question: str, retrieved_faq: Dict[str, str]) -> str:
    context = (
        f"FAQ Question: {retrieved_faq['question']}\n"
        f"FAQ Answer: {retrieved_faq['answer']}"
    )

    chat_completion = client_faq.chat.completions.create(
        messages=[
            {"role": "system", "content": faq_prompt},
            {
                "role": "user",
                "content": f"USER QUESTION: {question}\n\nFAQ CONTEXT:\n{context}",
            },
        ],
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=256,
    )

    return chat_completion.choices[0].message.content


def faq_chain(question: str, min_score: float = 0.35) -> str:
    matches = _retrieve_faq(question, top_k=1)
    if not matches:
        return "FAQ data is not loaded yet."

    best_match = matches[0]
    if best_match["score"] < min_score:
        return (
            "I could not find a matching FAQ for that. "
            "Please rephrase your question or ask about returns, refunds, payments, tracking, or offers."
        )

    try:
        return _generate_faq_answer(question, best_match)
    except Exception:
        return best_match["answer"]


if __name__ == "__main__":
    faqs_path = Path(__file__).parent / "resources/faq_data.csv"
    ingest_faq_data(faqs_path)
    print(faq_chain("Can I pay with UPI?"))
