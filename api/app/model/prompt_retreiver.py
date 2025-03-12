import torch
from model.embeddings import Embedding
from model.model_transform import model_transform
from model.web import search_web

def retrieve_relevant_docs(query, top_k=3):
    model_name= "sentence-transformers/all-mpnet-base-v2"
    
    print("------model name-----")
    print(model_name)
    embedding_obj= Embedding(model_name)
    embedding_model=embedding_obj.get_model_embedding(model_name)
    print(embedding_model)
    print("-----")
    query_embedding = embedding_model.encode(query)
    print("----vector----")
    vector_db= embedding_obj.vector_db
    print(vector_db)
    return vector_db.similarity_search_by_vector(query_embedding, k=top_k)
    

def format_prompt(query):
    relevant_docs = retrieve_relevant_docs(query)
    print("----docs----")
    print(relevant_docs)
    
    context = "\n\n".join([doc.page_content for doc in relevant_docs]) or search_web(query)
 
    # Buscamos en Google si no hay documentos relevantes
  
    return  f"""Usa la siguiente información para responder de manera clara y precisa: {context}

    Pregunta del usuario: {query}
    """



def generate_response(query,):
    print("request query")
    print(query)
    prompt = format_prompt(query)
    model, tokenizer= model_transform()
     
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    model.to(inputs.input_ids.device)
    
    output = model.generate(**inputs, max_length=500, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)
    
    
    

def retrieve_relevant_docs(self, top_k=3):
    query_embedding = Embedding.embedding_model.encode(self)
    
    return  self.vector_db.similarity_search_by_vector(query_embedding, k=top_k)
        