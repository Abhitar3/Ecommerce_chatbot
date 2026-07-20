import streamlit as st
from router import router
from faq import ingest_faq_data,faq_chain
from sql import sql_chain
from pathlib import Path
import re

faqs_path=Path(__file__).parent/'resources/faq_data.csv'
ingest_faq_data(faqs_path)

def keyword_route(query):
    q = re.sub(r"\s+", " ", query.lower()).strip()

    if "best deal" in q or "best deals" in q:
        return "sql"

    product_keywords = [
        "product",
        "products",
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
        "sandals",
        "loafers",
        "heels",
        "price",
        "discount",
        "sale",
        "sales",
        "brand",
        "rating",
        "under",
        "between",
        "list",
        "show",
        "top",
        "cheapest",
        "in stock",
    ]

    high_confidence_faq = [
        "refund",
        "return",
        "policy",
        "payment",
        "pay",
        "paid",
        "upi",
        "cash",
        "cash on delivery",
        "cod",
        "card",
        "net banking",
        "track",
        "order status",
        "damaged",
        "defective",
        "faulty",
        "cancel",
        "shipping",
        "coupon",
        "promo code",
    ]

    if any(keyword in q for keyword in high_confidence_faq):
        return "faq"

    faq_keywords = ["offer", "offers", "deal", "deals", "promo"]
    faq_score = sum(1 for keyword in faq_keywords if keyword in q)
    sql_score = sum(1 for keyword in product_keywords if keyword in q)

    if faq_score > sql_score:
        return "faq"
    if sql_score > faq_score:
        return "sql"
    return None

def ask(query):
    route = keyword_route(query) or router(query).name

    # Fallback routing when semantic router is uncertain (route can be None).
    if route is None:
        route = "faq"

    if route=='faq':
        return faq_chain(query)
    elif route=='sql':
        return sql_chain(query)
    else:
        return f"Route {route} not implemented yet"






st.title("E COMMERCE CHATBOT")

st.caption("Try a sample question or type your own.")

example_questions = [
    "How can I pay?",
    "What is the return policy?",
    "Show Nike shoes under 3000",
    "What are the best deals?",
    "How many Nike products do we have?",
    "What if my product is damaged?",
]

selected_example = None
example_cols = st.columns(2)
for idx, example in enumerate(example_questions):
    with example_cols[idx % 2]:
        if st.button(example, key=f"example_{idx}", use_container_width=True):
            selected_example = example

query = selected_example or st.chat_input("Write your query")


if 'messages' not in st.session_state:
    st.session_state['messages']=[]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({'role':'user','content':query})
    response=ask(query)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({'role':'assistant','content':response})
