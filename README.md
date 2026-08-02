# E-Commerce Chatbot with Multi-Intent Classification

AI-powered e-commerce chatbot that supports customer FAQ handling and natural-language product search over scraped Flipkart product data.

Live demo: https://abhitar-ecommerce-chatbot.streamlit.app  

## Overview

This project demonstrates a two-route chatbot architecture:

- `FAQ route`: retrieves relevant policy/support answers using hybrid RAG with FAISS dense search, BM25 sparse search, Reciprocal Rank Fusion, and cross-encoder reranking.
- `SQL route`: converts product-search questions into SQL using Groq, executes the query against SQLite, and returns product results with links.

Product data is collected offline using Selenium, saved to CSV, and loaded into SQLite. The live app does not scrape Flipkart during query time, so some external product links may expire or become unavailable.

## Features

- Multi-intent routing for FAQ and product-search queries.
- Hybrid FAQ retrieval using FAISS dense search, BM25 sparse search, Reciprocal Rank Fusion, and cross-encoder reranking.
- Groq-powered natural-language-to-SQL generation for product questions.
- SQLite product catalog built from scraped Flipkart data.
- Streamlit chat interface with sample prompt buttons.
- Fallback handling for unmatched intents, large SQL outputs, and unavailable exact product matches.
- Cleaner product responses with compact `View product` links.

## Architecture

```text
User query
-> keyword guardrails + semantic-router
-> FAQ route or SQL route
```

FAQ route:

```text
faq_data.csv
-> sentence-transformer embeddings
-> FAISS vector index
-> BM25 sparse index
-> dense + sparse retrieval
-> Reciprocal Rank Fusion
-> cross-encoder reranking
-> retrieve top FAQ contexts
-> Groq generates grounded final answer
```

Product route:

```text
User product query
-> Groq NL-to-SQL
-> SQLite query execution
-> product records
-> formatted response with product links
```

Data pipeline:

```text
Selenium scraping
-> Flipkart product CSV
-> CSV-to-SQL ingestion
-> SQLite database
-> Streamlit app queries SQLite at runtime
```

## Evaluation

Internal evaluation results:

- Intent routing accuracy: `82.5%`
- Macro-F1: `0.80`
- FAQ answer accuracy: `90%`
- Hybrid FAQ retrieval Top-1 accuracy: `93.5%`
- Hybrid FAQ retrieval Recall@3: `98.4%`
- Hybrid FAQ retrieval MRR: `0.96`

The evaluation set included paraphrased, noisy, shorthand, and ecommerce-specific user queries for FAQ and product-search intents. The FAQ retrieval benchmark contains `62` labeled queries.

Run FAQ retrieval evaluation:

```powershell
python app/evaluate_faq_retrieval.py
```

The retrieval evaluator reports:

- Top-1 accuracy
- Recall@3
- Mean Reciprocal Rank (MRR)

## Tech Stack

- Python
- Streamlit
- Groq
- Semantic Router
- Sentence Transformers
- FAISS
- BM25
- Cross Encoder
- SQLite
- Selenium
- Pandas

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create `app/.env`:

```text
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

4. Run the app:

```powershell
streamlit run app/main.py
```

## Example Questions

- How can I pay?
- What is the return policy?
- What if my product is damaged?
- Show Nike shoes under 3000
- What are the best deals?
- How many Nike products do we have?

## Notes

- Product data is based on an offline Flipkart scrape.
- The app queries SQLite at runtime; Selenium is not used during live chat.
- Some Flipkart product links may expire after scraping.
