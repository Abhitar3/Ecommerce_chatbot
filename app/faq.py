import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


_FAQ_DATA: List[Tuple[str, str]] = []
_FAQ_BY_KEY: Dict[str, str] = {}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _infer_key_from_question(question: str) -> str:
    q = _normalize(question)
    if "hdfc" in q:
        return "hdfc_discount"
    if "return policy" in q:
        return "return_policy"
    if "refund" in q:
        return "refund_time"
    if "track" in q:
        return "order_tracking"
    if "payment" in q:
        return "payment_methods"
    if "ongoing sales" in q or "promotions" in q:
        return "ongoing_offers"
    if "cancel" in q or "modify" in q:
        return "cancel_modify"
    if "international shipping" in q:
        return "international_shipping"
    if "damaged" in q:
        return "damaged_product"
    if "promo code" in q:
        return "promo_code"
    return ""


def ingest_faq_data(csv_path: Path) -> None:
    """Load FAQ data from CSV into process memory.

    Expected CSV columns: question, answer
    """
    global _FAQ_DATA, _FAQ_BY_KEY

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

    _FAQ_DATA = [
        (row.question, row.answer)
        for row in normalized_df.itertuples(index=False)
        if row.question and row.answer
    ]

    _FAQ_BY_KEY = {}
    for faq_question, faq_answer in _FAQ_DATA:
        key = _infer_key_from_question(faq_question)
        if key and key not in _FAQ_BY_KEY:
            _FAQ_BY_KEY[key] = faq_answer


def _match_rule_based(query: str) -> str:
    q = _normalize(query)

    if any(token in q for token in ["upi", "payment", "pay", "card", "cash on delivery", "net banking"]):
        return _FAQ_BY_KEY.get("payment_methods", "")

    if any(token in q for token in ["track", "where is my order", "order status", "shipped", "delivery status"]):
        return _FAQ_BY_KEY.get("order_tracking", "")

    if "hdfc" in q and any(token in q for token in ["discount", "offer", "deal"]):
        return _FAQ_BY_KEY.get("hdfc_discount", "")

    if any(token in q for token in ["offer", "offers", "deal", "deals", "sale", "promo"]):
        return _FAQ_BY_KEY.get("ongoing_offers", "")

    if any(token in q for token in ["damaged", "faulty", "defective", "broken"]):
        return _FAQ_BY_KEY.get("damaged_product", "")

    if any(token in q for token in ["return policy", "return product", "return item", "return"]):
        return _FAQ_BY_KEY.get("return_policy", "")

    if any(token in q for token in ["refund", "money back"]):
        return _FAQ_BY_KEY.get("refund_time", "")

    if any(token in q for token in ["cancel", "modify", "change order"]):
        return _FAQ_BY_KEY.get("cancel_modify", "")

    if any(token in q for token in ["international shipping", "ship internationally", "ship abroad", "outside india"]):
        return _FAQ_BY_KEY.get("international_shipping", "")

    if any(token in q for token in ["promo code", "coupon", "coupon code"]):
        return _FAQ_BY_KEY.get("promo_code", "")

    return ""


def _match_fuzzy(query: str, min_score: float = 0.55) -> str:
    if not _FAQ_DATA:
        return ""

    query_norm = _normalize(query)

    best_answer = ""
    best_score = -1.0

    for faq_question, faq_answer in _FAQ_DATA:
        score = SequenceMatcher(None, query_norm, _normalize(faq_question)).ratio()
        if score > best_score:
            best_score = score
            best_answer = faq_answer

    if best_score >= min_score:
        return best_answer

    return ""


def faq_chain(question: str) -> str:
    if not _FAQ_DATA:
        return "FAQ data is not loaded yet."

    rule_answer = _match_rule_based(question)
    if rule_answer:
        return rule_answer

    fuzzy_answer = _match_fuzzy(question)
    if fuzzy_answer:
        return fuzzy_answer

    return (
        "I could not find a matching FAQ for that. "
        "Please rephrase your question or ask about returns, refunds, payments, tracking, or offers."
    )


if __name__ == "__main__":
    faqs_path = Path(__file__).parent / "resources/faq_data.csv"
    ingest_faq_data(faqs_path)
    print(faq_chain("Can I pay with UPI?"))
