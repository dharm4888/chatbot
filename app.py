import os
import re
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Updated LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment variables.")

# Set it in the environment so OpenAI libs can find it
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:1234@127.0.0.1:3306/altiq_tshirts")

# Then just use:
embeddings = OpenAIEmbeddings()

try:
    faiss_index = FAISS.load_local(
        "faiss_index_docs",
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index: {e}")

retriever = faiss_index.as_retriever(search_kwargs={"k": 5})

# -----------------------------
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7
)
PROMPT = PromptTemplate.from_template(
    """You are an assistant that answers user questions using the given context.

Context:
{context}

Question: {question}

Answer concisely and cite sources when possible.
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# -----------------------------
# 4. SQL tool (safe SELECT only)
# -----------------------------
engine = create_engine(DATABASE_URL)
ALLOWED_TABLES = {"t_shirts", "discounts"}


def run_safe_sql(query: str) -> List[Dict]:
    """
    Execute a safe SELECT query, validating for allowed tables and no multi-statements.
    """
    query_stripped = query.strip().lower()
    if not query_stripped.startswith("select"):
        raise HTTPException(status_code=403, detail="Only SELECT queries are allowed.")
    if ";" in query:
        raise HTTPException(status_code=403, detail="Multiple statements not allowed.")

    # Extract tables from FROM and JOIN clauses (case-insensitive)
    table_matches = re.findall(r'\b(?:from|join)\s+(\w+)', query, re.IGNORECASE)
    for table in table_matches:
        if table.lower() not in ALLOWED_TABLES:
            raise HTTPException(status_code=403, detail=f"Query references disallowed table: {table}")

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
        return [dict(row._mapping) for row in result]
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


# -----------------------------
# 5. FastAPI app
# -----------------------------
app = FastAPI(title="RAG + SQL Assistant API")

class ChatRequest(BaseModel):
    message: str
    use_sql_tool: bool = False
    sql_query: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    sql_result: Optional[List[Dict]] = None

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sql_result = None

    if req.use_sql_tool:
        if not req.sql_query:
            raise HTTPException(
                status_code=400,
                detail="sql_query is required when use_sql_tool=True"
            )
        sql_result = run_safe_sql(req.sql_query)

    try:
        output = qa_chain({"question": req.message})
        answer = output["result"]
        sources = [doc.metadata.get("source", "unknown") for doc in output["source_documents"]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {e}")

    return ChatResponse(answer=answer, sources=sources, sql_result=sql_result)