import fitz  # PyMuPDF propuesta
import os


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