from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import json
from model.pdf_text import text_extract
import os

#class Embedding:

 #   embedding_model= None
  #  vector_db= None
 

   # def __init__(self, model_name):
    #  self.embedding_model = self.get_model_embedding(model_name)
     # self.vector_db= self.vector_to_model_embeddings


def get_model_embedding( model_name):
    return  SentenceTransformer(model_name)

def build_embeddings_policies_db():
    
    if not os.path.exists("insurance_policies_db"):
        return "Embeddings already generated"
        
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





def build_embeddings_qna():

    # Nuestro Modelo de embeddings
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Cargamos las preguntas y respuestas que hicimos en equipo el documento Q&A
    qa_file = "qna/qna_polizas.json"  # Cambia esto por la ubicación real del archivo

    with open(qa_file, "r", encoding="utf-8") as file:
        qna_data = json.load(file)  # Formato esperado: [{"question": "...", "answer": "..."}, ...]

    # Generamos los embeddings para cada pregunta
    questions = [qa["question"] for qa in qna_data]
    answers = [qa["answer"] for qa in qna_data]

    # Convertimos las preguntas a embeddings
    question_embeddings = embedding_model.encode(questions)

    # Crear el documento de LangChain
    docs = [Document(page_content=answers[i], metadata={"question": questions[i]}) for i in range(len(questions))]

    # Lo almacenamos en FAISS
    qna_db = FAISS.from_embeddings(
        [(doc.page_content, emb) for doc, emb in zip(docs, question_embeddings)],
        embedding_model
    )

    # Guardamos el doc generado de FAISS en disco
    qna_db.save_local("insurance_qna_db")

    print("✅ Embeddings de Base de datos FAISS con Q&A creada.")

    
    
def vector_to_model_embeddings():
        
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Cargamos la base de datos FAISS
    vector_db_policies = FAISS.load_local("insurance_policies_db", embedding_model, allow_dangerous_deserialization=True,)
    # Cargarmos la base de datos de Q&A propia
    vector_db_qna = FAISS.load_local("insurance_qna_db", embedding_model, allow_dangerous_deserialization=True)

    # Combinamos ambas bases FAISS en una sola
    vector_db_policies.merge_from(vector_db_qna)

    # Ahora vector_db contiene ambos conjuntos de datos
    vector_db = vector_db_policies

    print("✅ FAISS con documentos y Q&A cargado correctamente.")
    
    return vector_db
