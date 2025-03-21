import fitz  # PyMuPDF propuesta
import os
from model.preprocessing import convert_files_to_docs, process_documents
import pandas as pd

def extract_text_from_pdf(pdf_path):
    """ Extrae texto de un archivo PDF """
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])
    


# Extraemos el texto de todos los PDFs
def text_extract():
    pdf_folder = os.path.dirname("./dataset/") ### Vamos a extraer el texto de los PDFs
    pdf_texts = {}
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            pdf_texts[file] = extract_text_from_pdf(pdf_path)

    # Text example
    for pdf, text in pdf_texts.items():
        print(f"\n📄 {pdf} (Primeros 500 caracteres):\n{text[:500]}")
        break
    
    return pdf_texts


def process_docs():
    dir_path = os.getcwd()+"/model/docs/dataset/"
    all_docs = convert_files_to_docs(dir_path)
    print("----------docs---------- ")
    print(dir_path)
    docs_default, processed_docs = process_documents(all_docs)
    return docs_default, processed_docs, all_docs


def build_text_dataframe():
    docs_default, processed_docs = process_docs()
    df = text_to_dataframe(docs_default)
    return df

def text_to_dataframe(docs_default):
    # Extraer el contenido de los documentos en una lista de textos usando la clave content
    texts = [doc["content"] for doc in docs_default] 

    # Crear DataFrame con los textos
    df = pd.DataFrame({"text": texts})

    # Agregar estadísticas de texto
    df["num_words"] = df["text"].apply(lambda x: len(x.split()))
    df["num_chars"] = df["text"].apply(lambda x: len(x))

    # Mostrar estadísticas generales
    print(df[["num_words", "num_chars"]].describe())  # Muestra distribución
    print(f"Total words: {df['num_words'].sum()}")  # Suma total de palabras
    print(f"Total chars: {df['num_chars'].sum()}")  # Suma total de caracteres

    return df

