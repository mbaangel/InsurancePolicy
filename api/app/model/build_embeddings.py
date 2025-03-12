from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import json
from pdf_text import text_extract
import os



async  def build_embeddings_policies_db():
    
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # Dividimos texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    pdf_texts=text_extract
    docs = []
    for pdf_name, text in pdf_texts.items():
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"source": pdf_name}))

    # Convertir a embeddings y almacenar en FAISS
    texts = [doc.page_content for doc in docs]
    embeddings = embedding_model.encode(texts)

    vector_db = FAISS.from_embeddings(
        [(doc.page_content, emb) for doc, emb in zip(docs, embeddings)],
        embedding_model
    )

    vector_db.save_local("insurance_policies_db")

    return "✅ Embeddings generated and stored in FAISS."