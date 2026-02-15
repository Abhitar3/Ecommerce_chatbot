from semantic_router import Route
from semantic_router.routers import SemanticRouter as RouteLayer
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.index import LocalIndex

# Step 1: Define the encoder
encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Define the routes

faq = Route(
    name='faq',
    utterances=[
        # Policy-related
        "What is the return policy of the products?",
        "what is your refund policy",
        "How long does it take to process a refund?",
        "Can I return items after 30 days?",
        "What's the process to return a damaged item?",
        "What is your policy on defective items?",
        "How do I report a faulty product?",
        "Do you accept returns for faulty products?",
        "What happens if a product is damaged?",
        "Is there a warranty on electronics?",
        "How can I claim a warranty?",

        # Payment-related
        "What payment methods are accepted?",
        "Can I pay with UPI?",
        "Do you accept cash on delivery?",
        "Do you take American Express?",
        "Can I use Paytm for checkout?",
        "Do you accept international cards?",
        "Can I use net banking?",

        # Tracking
        "How can I track my order?",
        "Where is my package?",
        "Order status please?",
        "How do I know if my order has shipped?",
        "Can I see the delivery estimate?",
        "What is the expected delivery date?",

        # Discounts / Offers
        "Do I get discount with the HDFC credit card?",
        "Any current offers or deals?",
        "Do you have seasonal discounts?",
        "Are there promo codes available?",
        "Any discount on first order?",
    ]
)

sql = Route(
    name='sql',
    utterances=[
        # Product queries
        "I want to buy nike shoes that have 50% discount.",
        "Are there any shoes under Rs. 3000?",
        "Do you have formal shoes in size 9?",
        "What is the price of puma running shoes?",
        "Show me adidas sneakers below 2000",
        "List black sports shoes in size 10",
        "Any men's loafers available in brown?",
        "I'm looking for women's sandals under ₹1500",
        "Display Reebok shoes on sale",
        "Which brands are offering discounts today?",
        "What are the cheapest Puma shoes you have?",
        "Are there any shoes in the 1000 to 2000 range?",

        # Filtering queries
        "Pink Puma shoes in price range 5000 to 1000",
        "Show me blue Nike sneakers under ₹2500",
        "List size 8 formal black shoes under 3000",
        "Search women's heels between ₹1200 and ₹2500",
        "I want white sneakers under 2000",
        "Which shoes are in stock for size 10 and red color?",
    ]
)

# Step 3: Create the index
index = LocalIndex()

# Step 4: Initialize router with index and auto_sync
router = RouteLayer(
    encoder=encoder,
    routes=[faq, sql],
    index=index,
    auto_sync="local"
)

# Step 5: Use the router
if __name__ == "__main__":
    print(router("What is your policy on defective product?").name)  # → faq
    print(router("Pink Puma shoes in price range 5000 to 1000").name)  # → sql
    print(router("Do you offer cash on delivery?").name)  # → faq
    print(router("Cheapest red sneakers in size 9?").name)  # → sql
