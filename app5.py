import os
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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
st.sidebar.title("Settings")
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
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
   ## api_key=OPENAI_API_KEY
)

# -----------------------------
# 4. SQL helper (safe SELECT only)
# -----------------------------
engine = create_engine(DATABASE_URL)
ALLOWED_TABLES = {"t_shirts", "discounts"}  # whitelist

def run_safe_sql(query: str):
    q = query.strip().lower()
    if not q.startswith("select"):
        st.error("❌ Only SELECT queries are allowed.")
        return None
    if not any(tbl in q for tbl in ALLOWED_TABLES):
        st.error("❌ Query references disallowed tables.")
        return None

    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
    return [dict(row._mapping) for row in result]

# -----------------------------
# 5. Streamlit UI
# -----------------------------
st.title("🧑‍💻 Chatbot with RAG + SQL Tool")

user_message = st.text_area("💬 Ask your question:")
use_sql = st.checkbox("🔍 Use SQL tool")
sql_query = None

if use_sql:
    sql_query = st.text_input("Enter SQL query (SELECT only):")

if st.button("Submit"):
    sql_result = None
    if use_sql and sql_query:
        sql_result = run_safe_sql(sql_query)

    # Retrieve relevant docs
    docs = retriever.get_relevant_documents(user_message)

    if docs:
        context = "\n\n".join([d.page_content for d in docs])
        answer = llm.invoke(
            f"Answer the question using ONLY the following context.\n"
            f"If the context does not contain the answer, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\nQuestion: {user_message}"
        ).content
        sources = [d.metadata.get("source", "unknown") for d in docs]

        # fallback if "I don't know"
        if "i don't know" in answer.lower(): # type: ignore
            answer = llm.invoke(user_message).content
            sources = []
    else:
        answer = llm.invoke(user_message).content
        sources = []

    # -----------------------------
    # 6. Show results
    # -----------------------------
    st.subheader("🤖 Answer")
    st.write(answer)

    if sources:
        st.subheader("📚 Sources")
        for s in sources:
            st.write(f"- {s}")

    if sql_result:
        st.subheader("📊 SQL Results")
        st.dataframe(sql_result)
