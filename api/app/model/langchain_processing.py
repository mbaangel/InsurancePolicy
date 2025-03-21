from langchain_core.documents import Document
from langchain.vectorstores import FAISS

def get_langchain_docs(docs_default):
    # Convertir diccionarios a objetos Document de LangChain
    langchain_docs = [
        Document(page_content=doc["content"], metadata=doc.get("meta", {}))  
        for doc in docs_default
    ]
    return langchain_docs




def vector_store(embeddings_model, langchain_docs):
    return FAISS.from_documents(langchain_docs, embeddings_model)
