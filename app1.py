import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from typing import Optional

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = "mysql+pymysql://root:1234@127.0.0.1:3306/altiq_tshirts"

# -----------------------------
# 2. LangChain imports
# -----------------------------
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# -----------------------------
# 3. Load FAISS retriever
# -----------------------------
embeddings = OpenAIEmbeddings()

faiss_index = FAISS.load_local(
    "faiss_index_docs",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = faiss_index.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# 4. LLM model
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    
)

# -----------------------------
# 5. SQL tool (safe SELECT only)
# -----------------------------
engine = create_engine(DATABASE_URL)
ALLOWED_TABLES = {"t_shirts", "discounts"}  # whitelist

def run_safe_sql(query: str):
    q = query.strip().lower()
    if not q.startswith("select"):
        raise HTTPException(status_code=403, detail="Only SELECT queries are allowed.")
    if not any(tbl in q for tbl in ALLOWED_TABLES):
        raise HTTPException(status_code=403, detail="Query references disallowed tables.")

    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
    return [dict(row._mapping) for row in result]

# -----------------------------
# 6. FastAPI app
# -----------------------------
app1 = FastAPI()

class ChatRequest(BaseModel):
    message: str
    use_sql_tool: bool = False
    sql_query: Optional[str] = None

@app1.post("/chat")
def chat(req: ChatRequest):
    sql_result = None
    if req.use_sql_tool:
        if not req.sql_query:
            raise HTTPException(status_code=400, detail="sql_query is required when use_sql_tool=True")
        sql_result = run_safe_sql(req.sql_query)

    # Retrieve relevant documents from FAISS
    docs = retriever.get_relevant_documents(req.message)

    if docs:
        # RAG: answer using documents
        context = "\n\n".join([d.page_content for d in docs])
        answer = llm.invoke(
            f"Answer the question using ONLY the following context.\n"
            f"If the context does not contain the answer, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\nQuestion: {req.message}"
        ).content
        sources = [d.metadata.get("source", "unknown") for d in docs]

        # Fallback if answer indicates missing info
        if "i don't know" in answer.lower() or "does not provide information" in answer.lower(): # type: ignore
            answer = llm.invoke(req.message).content
            sources = []
    else:
        # No documents retrieved, fallback to general knowledge LLM
        answer = llm.invoke(req.message).content
        sources = []

    return {
        "answer": answer,
        "sources": sources,
        "sql_result": sql_result
    }
