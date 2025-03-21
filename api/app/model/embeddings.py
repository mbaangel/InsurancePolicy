from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import json
import os

def get_hugging_face_model_embedding( ):
    embeddings_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2',
    )
    return embeddings_model    


def get_model_embedding( model_name):
    return  SentenceTransformer(model_name)
