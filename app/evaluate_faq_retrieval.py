from pathlib import Path

from faq import ingest_faq_data, retrieve_faq


EVAL_DATA = [
    {
        "query": "how can i pay",
        "expected": "What payment methods are accepted?",
    },
    {
        "query": "can i use upi or card",
        "expected": "What payment methods are accepted?",
    },
    {
        "query": "how many days for refund",
        "expected": "How long does it take to process a refund?",
    },
    {
        "query": "where is my parcel",
        "expected": "How can I track my order?",
    },
    {
        "query": "what if my product is broken",
        "expected": "What should I do if I receive a damaged product?",
    },
    {
        "query": "can i cancel after ordering",
        "expected": "Can I cancel or modify my order after placing it?",
    },
    {
        "query": "do you ship outside india",
        "expected": "Do you offer international shipping?",
    },
    {
        "query": "how do i apply coupon",
        "expected": "How do I use a promo code during checkout?",
    },
    {
        "query": "any current deals",
        "expected": "Are there any ongoing sales or promotions?",
    },
    {
        "query": "hdfc card discount",
        "expected": "Do I get discount with the HDFC credit card?",
    },
    {
        "query": "return window for products",
        "expected": "What is the return policy of the products?",
    },
    {
        "query": "how to track shipping status",
        "expected": "How can I track my order?",
    },
]


def evaluate():
    faqs_path = Path(__file__).parent / "resources" / "faq_data.csv"
    ingest_faq_data(faqs_path)

    top_1_hits = 0
    recall_at_3_hits = 0
    reciprocal_ranks = []

    for item in EVAL_DATA:
        results = retrieve_faq(item["query"], top_k=3)
        retrieved_questions = [result["question"] for result in results]

        if retrieved_questions and retrieved_questions[0] == item["expected"]:
            top_1_hits += 1

        if item["expected"] in retrieved_questions:
            recall_at_3_hits += 1
            reciprocal_ranks.append(1 / (retrieved_questions.index(item["expected"]) + 1))
        else:
            reciprocal_ranks.append(0)

        print(f"Query: {item['query']}")
        print(f"Expected: {item['expected']}")
        print(f"Retrieved: {retrieved_questions}")
        print()

    total = len(EVAL_DATA)
    top_1_accuracy = top_1_hits / total
    recall_at_3 = recall_at_3_hits / total
    mrr = sum(reciprocal_ranks) / total

    print("FAQ RETRIEVAL EVALUATION")
    print(f"Total queries: {total}")
    print(f"Top-1 accuracy: {top_1_accuracy:.3f}")
    print(f"Recall@3: {recall_at_3:.3f}")
    print(f"MRR: {mrr:.3f}")


if __name__ == "__main__":
    evaluate()
