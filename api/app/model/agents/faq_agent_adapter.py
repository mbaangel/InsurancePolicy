from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from model.embeddings import get_hugging_face_model_embedding
from langchain.vectorstores import FAISS
from model.large_langueje_model import get_large_language_model
from model.utils import read_qna_data

faq_template = """Eres un asistente especializado en pólizas de seguros de salud.
Responde con precisión según la información dada. Responde SIEMPRE en el idioma en el que se hizo la pregunta. Usa el historial de la conversación para generar respuestas más personalizadas.
Responde de manera clara y concisa, sin agregar demasiada información irrelevante. 
Si la respuesta no está en el contexto u obtienes una respuesta irrelevante, o si la pregunta está relacionada con noticias, actualidad o si el usuario te saluda, 
responde con "No puedo encontrar información sobre esto en mis FAQs".



Contexto: {context}

Pregunta: {question}
Respuesta:"""
llm= get_large_language_model()
qa_prompt = PromptTemplate(template=faq_template, input_variables=["context", "question"])
faq_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)

qna_data= read_qna_data()

qa_docs = [Document(page_content=qa["answer"], metadata={"question": qa["question"]}) for qa in qna_data]
qna_db = FAISS.from_documents(qa_docs, get_hugging_face_model_embedding())
qna_db.save_local("insurance_qna_db")






def qna_search(search_input, threshold=0.75):
    """Busca en la base de datos de FAQs los documentos más similares a la pregunta."""
    qa_docs = qna_db.similarity_search_with_score(search_input, k=5)  # Usamos más documentos para tener más opciones
    sorted_docs = sorted(qa_docs, key=lambda x: x[1])  # Ordenar por puntuación de similitud
    # Filtrar solo los documentos con puntuación debajo del umbral
    filtered_docs = [doc[0] for doc in sorted_docs if doc[1] < threshold]
    return filtered_docs



def faq_agent_original(question):
    """Searches for answers in the FAQs database or passes to the next agent if no relevant matches."""
    
    docs = qna_search(question, threshold=0.75)
    
    if not docs:
        return {"response": "Pasar al siguiente agente"}

    most_relevant_doc = docs[0]
    combined_context = most_relevant_doc.page_content

    try:
        prompt = f"Contexto: {combined_context}\nPregunta: {question}\nRespuesta:"
        direct_response = llm.invoke(prompt)

        if direct_response and "no puedo encontrar información" not in direct_response.lower():
            response_start = direct_response.lower().find("respuesta:")
            if response_start != -1:
                answer = direct_response[response_start + len("Respuesta:"):].strip()

            else:
                answer = direct_response.strip()

                if "Explicación:" in answer:
                    answer = answer.split("Explicación:")[0].strip()

                return {"response": answer.split("\n")[0].strip()}
           # return {"response": direct_response}
    except Exception as e:
        print(f"Error en la aproximación directa: {e}")

    try:
        response = faq_chain({"input_documents": [most_relevant_doc], "question": question})
        output_text = response.get("output_text", response).strip()
        final_answer = output_text.split("Respuesta:")[-1].strip()

        if "Explicación:" in final_answer:
            final_answer = final_answer.split("Explicación:")[0].strip()

        return {"response": final_answer.split("\n")[0].strip()} if final_answer else {"response": "Pasar al siguiente agente"}
    except Exception as e:
        print(f"Error en faq_chain: {e}")
        return {"response": "Pasar al siguiente agente"}



def faq_agent_adapter(state):
    """Adapts the input/output format of faq_agent_original to work with LangGraph."""

    question = state.get("question", "")
    #print(f"FAQ Agent recibiendo: {question}")  # Debug
    
    result = faq_agent_original(question)
    #print(f"FAQ Agent resultado: {result}")  # Debug
    
    new_state = state.copy()
    if isinstance(result, dict) and "response" in result:
        new_state["response"] = result["response"]
        #if result["response"] != "Pasar al siguiente agente":
            #print(f"FAQ Agent encontró respuesta: {result['response']}")
            #print(f"{result['response']}")
    
    return new_state