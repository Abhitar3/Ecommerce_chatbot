from pathlib import Path

from faq import ingest_faq_data, retrieve_faq


EVAL_DATA = [
    # Payment methods
    {"query": "how can i pay", "expected": "What payment methods are accepted?"},
    {"query": "can i use upi or card", "expected": "What payment methods are accepted?"},
    {"query": "do you accept cash", "expected": "What payment methods are accepted?"},
    {"query": "cod available", "expected": "What payment methods are accepted?"},
    {"query": "payment options pls", "expected": "What payment methods are accepted?"},
    {"query": "online payment methods", "expected": "What payment methods are accepted?"},
    {"query": "can i pay with net banking", "expected": "What payment methods are accepted?"},
    {"query": "do u take debit card", "expected": "What payment methods are accepted?"},

    # Returns
    {"query": "return window for products", "expected": "What is the return policy of the products?"},
    {"query": "what is return policy", "expected": "What is the return policy of the products?"},
    {"query": "how many days do i have to return", "expected": "What is the return policy of the products?"},
    {"query": "can i send back an item", "expected": "What is the return policy of the products?"},
    {"query": "return period after delivery", "expected": "What is the return policy of the products?"},
    {"query": "product return rules", "expected": "What is the return policy of the products?"},

    # Refunds
    {"query": "how many days for refund", "expected": "How long does it take to process a refund?"},
    {"query": "when will refund come", "expected": "How long does it take to process a refund?"},
    {"query": "refund processing time", "expected": "How long does it take to process a refund?"},
    {"query": "how long for money back", "expected": "How long does it take to process a refund?"},
    {"query": "refund kab milega", "expected": "How long does it take to process a refund?"},
    {"query": "after return when do i get paid back", "expected": "How long does it take to process a refund?"},

    # Tracking
    {"query": "where is my parcel", "expected": "How can I track my order?"},
    {"query": "how to track shipping status", "expected": "How can I track my order?"},
    {"query": "package status", "expected": "How can I track my order?"},
    {"query": "has my order shipped", "expected": "How can I track my order?"},
    {"query": "track my delivery", "expected": "How can I track my order?"},
    {"query": "where can i see order status", "expected": "How can I track my order?"},

    # Damaged or defective product
    {"query": "what if my product is broken", "expected": "What should I do if I receive a damaged product?"},
    {"query": "damaged product received", "expected": "What should I do if I receive a damaged product?"},
    {"query": "item arrived faulty", "expected": "What should I do if I receive a damaged product?"},
    {"query": "received defective shoes", "expected": "What should I do if I receive a damaged product?"},
    {"query": "product came damaged what now", "expected": "What should I do if I receive a damaged product?"},
    {"query": "replacement for broken item", "expected": "What should I do if I receive a damaged product?"},

    # Cancellation or modification
    {"query": "can i cancel after ordering", "expected": "Can I cancel or modify my order after placing it?"},
    {"query": "modify order after placing", "expected": "Can I cancel or modify my order after placing it?"},
    {"query": "change my order details", "expected": "Can I cancel or modify my order after placing it?"},
    {"query": "cancel order immediately", "expected": "Can I cancel or modify my order after placing it?"},
    {"query": "can i update order after checkout", "expected": "Can I cancel or modify my order after placing it?"},
    {"query": "order edit allowed", "expected": "Can I cancel or modify my order after placing it?"},

    # International shipping
    {"query": "do you ship outside india", "expected": "Do you offer international shipping?"},
    {"query": "international delivery available", "expected": "Do you offer international shipping?"},
    {"query": "ship abroad?", "expected": "Do you offer international shipping?"},
    {"query": "can you deliver internationally", "expected": "Do you offer international shipping?"},
    {"query": "shipping to other countries", "expected": "Do you offer international shipping?"},
    {"query": "overseas shipping supported", "expected": "Do you offer international shipping?"},

    # Promo codes
    {"query": "how do i apply coupon", "expected": "How do I use a promo code during checkout?"},
    {"query": "where to enter promo code", "expected": "How do I use a promo code during checkout?"},
    {"query": "apply discount code at checkout", "expected": "How do I use a promo code during checkout?"},
    {"query": "coupon code use kaise kare", "expected": "How do I use a promo code during checkout?"},
    {"query": "promo field during payment", "expected": "How do I use a promo code during checkout?"},
    {"query": "where can i add voucher", "expected": "How do I use a promo code during checkout?"},

    # Offers and promotions
    {"query": "any current deals", "expected": "Are there any ongoing sales or promotions?"},
    {"query": "any offers today", "expected": "Are there any ongoing sales or promotions?"},
    {"query": "ongoing discount sale", "expected": "Are there any ongoing sales or promotions?"},
    {"query": "do you have promotions", "expected": "Are there any ongoing sales or promotions?"},
    {"query": "current sale offers", "expected": "Are there any ongoing sales or promotions?"},
    {"query": "deals section info", "expected": "Are there any ongoing sales or promotions?"},

    # HDFC discount
    {"query": "hdfc card discount", "expected": "Do I get discount with the HDFC credit card?"},
    {"query": "hdfc credit card offer", "expected": "Do I get discount with the HDFC credit card?"},
    {"query": "bank discount with hdfc", "expected": "Do I get discount with the HDFC credit card?"},
    {"query": "hdfc users get offer?", "expected": "Do I get discount with the HDFC credit card?"},
    {"query": "10 percent hdfc deal", "expected": "Do I get discount with the HDFC credit card?"},
    {"query": "credit card discount hdfc", "expected": "Do I get discount with the HDFC credit card?"},
]


def evaluate():
    faqs_path = Path(__file__).parent / "resources" / "faq_data.csv"
    ingest_faq_data(faqs_path)

    top_1_hits = 0
    recall_at_3_hits = 0
    reciprocal_ranks = []
    misses = []

    for item in EVAL_DATA:
        results = retrieve_faq(item["query"], top_k=3)
        retrieved_questions = [result["question"] for result in results]

        if retrieved_questions and retrieved_questions[0] == item["expected"]:
            top_1_hits += 1

        found_expected = item["expected"] in retrieved_questions
        if found_expected:
            recall_at_3_hits += 1
            reciprocal_ranks.append(1 / (retrieved_questions.index(item["expected"]) + 1))
        else:
            reciprocal_ranks.append(0)

        if not retrieved_questions or retrieved_questions[0] != item["expected"]:
            misses.append((item, retrieved_questions))

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

    if misses:
        print()
        print("MISSES")
        for item, retrieved_questions in misses:
            print(f"Query: {item['query']}")
            print(f"Expected: {item['expected']}")
            print(f"Retrieved: {retrieved_questions}")
            print()


if __name__ == "__main__":
    evaluate()
