import os
import re
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = "mysql+pymysql://root:1234@127.0.0.1:3306/altiq_tshirts"

# -----------------------------
# 2. Load FAISS retriever
# -----------------------------
embeddings = OpenAIEmbeddings()
faiss_index = FAISS.load_local(
    "faiss_index_docs",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = faiss_index.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# 3. LLM model
# -----------------------------
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# -----------------------------
# 4. SQL tool (safe SELECT only)
# -----------------------------
engine = create_engine(DATABASE_URL)
ALLOWED_TABLES = {"t_shirts", "discounts"}  

def run_safe_sql(query: str):
    q = query.strip().lower()
    if not q.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    if not any(tbl in q for tbl in ALLOWED_TABLES):
        raise ValueError("Query references disallowed tables.")
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
        return [dict(row._mapping) for row in result]
    except SQLAlchemyError as e:
        raise ValueError(f"Database error: {str(e)}")

# -----------------------------
# 5. Automatic SQL generator
# -----------------------------
def generate_sql_from_message(message: str):
    msg = message.lower()

    # Top N brands
    if "top" in msg and "t_shirts" in msg and "brand" in msg:
        n_match = re.search(r"top\s+(\d+)", msg)
        n = int(n_match.group(1)) if n_match else 5
        return f"SELECT brand, COUNT(*) as count FROM t_shirts GROUP BY brand ORDER BY count DESC LIMIT {n}"

    # Price of specific t-shirt
    if "price" in msg and "t_shirts" in msg:
        brand_match = re.search(r"price of (\w+)", msg)
        color_match = re.search(r"(black|white|red|blue|green|yellow|gray|pink)", msg)
        size_match = re.search(r"\b(s|m|l|xl|xxl)\b", msg)
        brand = brand_match.group(1) if brand_match else None
        color = color_match.group(1) if color_match else None
        size = size_match.group(1) if size_match else None

        conditions = []
        if brand:
            conditions.append(f"brand='{brand}'")
        if color:
            conditions.append(f"color='{color}'")
        if size:
            conditions.append(f"size='{size.upper()}'")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return f"SELECT brand, color, size, price FROM t_shirts WHERE {where_clause} LIMIT 1"

    return None

# -----------------------------
# 6. Flask app
# -----------------------------
app4 = Flask(__name__)

@app4.route("/", methods=["GET"])
def home():
    return {
        "message": "Chatbot API is running 🚀",
        "endpoints": {
            "GET /": "This home page",
            "POST /chat": "Chat with the assistant",
            "GET /ui": "Simple chatbot UI"
        }
    }

@app4.route("/ui", methods=["GET"])
def ui():
    return render_template("index.html")   # load from templates/index.html

@app4.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "message is required"}), 400

        message = data["message"]
        sql_result = None

        # Automatic SQL generation
        sql_query = generate_sql_from_message(message)
        if sql_query:
            sql_result = run_safe_sql(sql_query)

        # Retrieve relevant documents from FAISS
        docs = retriever.get_relevant_documents(message)
        if docs and len(docs) > 0:
            context = "\n\n".join([d.page_content for d in docs])
            answer = llm.invoke(
                f"Answer the question using ONLY the following context.\n"
                f"If the context does not contain the answer, say 'I don't know'.\n\n"
                f"Context:\n{context}\n\nQuestion: {message}"
            ).content
            sources = [d.metadata.get("source", "unknown") for d in docs]

            if answer and ("i don't know" in answer.lower() or "does not provide information" in answer.lower()): # type: ignore
                fallback = llm.invoke(message)
                answer = fallback.content if hasattr(fallback, "content") else str(fallback)
                sources = []
        else:
            # Fallback LLM
            answer = llm.invoke(message).content
            sources = []

        # Combine LLM answer + SQL result if available
        if sql_result:
            if len(sql_result) > 0:
                answer += "\n\nSQL Result:\n" + "\n".join([str(r) for r in sql_result])
            else:
                answer += "\n\nSQL Result: No data found."

        return jsonify({
            "answer": answer,
            "sources": sources,
            "sql_result": sql_result
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# -----------------------------
# Run Flask server
# -----------------------------
if __name__ == "__main__":
    app4.run(debug=True, host="0.0.0.0", port=8000)
