# file: ingest.py
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from sqlalchemy import create_engine, text
from langchain.schema import Document

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

DATABASE_URL = "mysql+pymysql://root:1234@127.0.0.1:3306/altiq_tshirts"

def load_pdfs(pdf_paths):
    docs = []
    for p in pdf_paths:
        loader = PyPDFLoader(p)
        pages = loader.load()
        docs.extend(pages)
    return docs

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def embed_and_save(docs, index_path="faiss_index"):
    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS.from_documents(docs, embeddings)
    faiss_index.save_local(index_path)
    print("Saved FAISS index to", index_path)
    return faiss_index

def ingest_sql_tables(table_names, index_path="faiss_index_sql"):
    engine = create_engine(DATABASE_URL)
    docs = []
    for table in table_names:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table} LIMIT 10000"))
            column_names = result.keys()   # ✅ get once outside the loop
            for r in result:
                text_blob = " | ".join(f"{c}: {v}" for c, v in zip(column_names, r))
                docs.append(Document(page_content=text_blob, metadata={"source": f"table:{table}"}))

    docs = chunk_documents(docs, chunk_size=800, chunk_overlap=100)
    return embed_and_save(docs, index_path)


if __name__ == "__main__":
    pdfs = ["DSML.pdf", "mcp.pdf","netact.pdf"]  # add your files
    docs = load_pdfs(pdfs)
    chunks = chunk_documents(docs)
    embed_and_save(chunks, index_path="faiss_index_docs")

    ingest_sql_tables(["t_shirts", "discounts"], index_path="faiss_index_sql")
