from groq import Groq, BadRequestError
import os
import re
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()

GROQ_MODEL = os.getenv('GROQ_MODEL')

db_path = Path(__file__).parent / "db.sqlite"

client_sql = Groq()

sql_prompt = """You are an expert in understanding the database schema and generating SQL queries for a natural language question asked
pertaining to the data you have. The schema is provided in the schema tags.
<schema>
table: product

fields:
product_link - string (hyperlink to product)
title - string (name of the product)
brand - string (brand of the product)
price - integer (price of the product in Indian Rupees)
discount - float (discount on the product. 10 percent discount is represented as 0.1, 20 percent as 0.2, and such.)
avg_rating - float (average rating of the product. Range 0-5, 5 is the highest.)
total_ratings - integer (total number of ratings for the product)

</schema>
Make sure whenever you try to search for the brand name, the name can be in any case.
So, make sure to use %LIKE% to find the brand in condition. Never use "ILIKE".
Create a single SQL query for the question provided.
The query should have all the fields in SELECT clause (i.e. SELECT *)

Just the SQL query is needed, nothing more. Always provide the SQL in between the <SQL></SQL> tags."""


comprehension_prompt = """You are an expert in understanding the context of the question and replying based on the data pertaining to the question provided. You will be provided with Question: and Data:. The data will be in the form of an array or a dataframe or dict. Reply based on only the data provided as Data for answering the question asked as Question. Do not write anything like 'Based on the data' or any other technical words. Just a plain simple natural language response.
The Data would always be in context to the question asked. For example if the question is "What is the average rating?" and data is "4.3", then answer should be "The average rating for the product is 4.3". So make sure the response is curated with the question and data. Make sure to note the column names to have some context, if needed, for your response.
There can also be cases where you are given an entire dataframe in the Data: field. Always remember that the data field contains the answer of the question asked. All you need to do is to always reply in the following format when asked about a product:
Product title, price in indian rupees, discount, and rating, and then product link. Take care that all the products are listed in list format, one line after the other. Not as a paragraph.
For example:
1. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>
2. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>
3. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>
"""


def generate_sql_query(question):
    chat_completion = client_sql.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": sql_prompt,
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1024,
    )

    return chat_completion.choices[0].message.content


def run_query(query):
    if query.strip().upper().startswith('SELECT'):
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn)
            return df
    return None


def data_comprehension(question, context):
    chat_completion = client_sql.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": comprehension_prompt,
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}. DATA: {context}",
            }
        ],
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=512,
    )

    return chat_completion.choices[0].message.content


def _compact_records(df: DataFrame, max_rows: int = 20):
    """Reduce payload size before sending SQL results to the LLM."""
    if df.empty:
        return []

    keep_cols = [
        "title",
        "brand",
        "price",
        "discount",
        "avg_rating",
        "total_ratings",
        "product_link",
    ]
    cols = [col for col in keep_cols if col in df.columns]
    compact_df = df[cols].copy().head(max_rows)

    if "title" in compact_df.columns:
        compact_df["title"] = compact_df["title"].astype(str).str.slice(0, 100)
    if "product_link" in compact_df.columns:
        compact_df["product_link"] = compact_df["product_link"].astype(str).str.slice(0, 180)

    return compact_df.to_dict(orient='records')


def _format_products_fallback(records):
    if not records:
        return "No matching products were found."

    lines = []
    for idx, row in enumerate(records, start=1):
        title = row.get("title", "Product")
        price = row.get("price", "N/A")
        discount = row.get("discount", "")
        rating = row.get("avg_rating", "N/A")
        link = row.get("product_link", "")

        if discount in ("", None) or pd.isna(discount):
            discount_text = "No discount info"
        else:
            try:
                discount_text = f"{int(float(discount) * 100)}% off"
            except Exception:
                discount_text = f"{discount} off"

        lines.append(f"{idx}. {title}: Rs. {price} ({discount_text}), Rating: {rating} {link}")

    return "\n".join(lines)


def sql_chain(question):
    sql_query = generate_sql_query(question)
    pattern = "<SQL>(.*?)</SQL>"
    matches = re.findall(pattern, sql_query, re.DOTALL)

    if len(matches) == 0:
        return "Sorry, LLM is not able to generate a query for your question"

    print(matches[0].strip())

    response = run_query(matches[0].strip())
    if response is None:
        return "Sorry, there was a problem executing SQL query"
    if response.empty:
        return "I could not find any matching products."

    context = _compact_records(response, max_rows=20)

    try:
        answer = data_comprehension(question, context)
        return answer
    except BadRequestError as e:
        if "context_length_exceeded" in str(e):
            return _format_products_fallback(context)
        raise


if __name__ == "__main__":
    question = "Show top 3 shoes in descending order of rating"
    answer = sql_chain(question)
    print(answer)
