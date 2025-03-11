import fitz  # PyMuPDF
from docx import Document
import os
import nltk
nltk.download('punkt', force=True)
print(nltk.data.path)

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format")
    
def convert_files_to_docs(dir_path):
    all_docs = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.pdf') or file.endswith('.docx'):
                text = extract_text_from_file(file_path)
                all_docs.append({"content": text, "meta": {"file_path": file_path}})
    return all_docs


def preprocess_text(text, split_length=100, split_respect_sentence_boundary=True):
    sentences = nltk.sent_tokenize(text)
    processed_docs = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > split_length and current_chunk:
            processed_docs.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        processed_docs.append(" ".join(current_chunk))

    return processed_docs

def process_documents(all_docs, split_length=100, split_respect_sentence_boundary=True):
    docs_default = []
    processed_docs = []

    for doc in all_docs:
        text = doc["content"]
        processed_parts = preprocess_text(text, split_length, split_respect_sentence_boundary)
        docs_default.extend([{"content": part, "meta": doc["meta"]} for part in processed_parts])
        processed_docs.append(processed_parts)

    print(f"n_docs_input: {len(all_docs)}")
    print(f"n_docs_output: {len(docs_default)}")

    return docs_default, processed_docs

