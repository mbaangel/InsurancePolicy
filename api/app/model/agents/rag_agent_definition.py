


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from model.large_langueje_model import get_large_language_model
from model.utils import clear_gpu_memory
from model.langchain_processing import  vector_store
from model.embeddings import get_hugging_face_model_embedding
from model.text_proccessing import process_docs
from model.langchain_processing import get_langchain_docs
from langchain.memory import ConversationBufferMemory


rag_template = """Eres un asistente especializado en pólizas de seguros de salud.
Responde con precisión según la información de los documentos proporcionados, usa el historial de la conversación para generar respuestas más personalizadas.
Responde de manera clara y concisa, sin agregar demasiada información irrelevante. 
Responde SIEMPRE en el idioma en el que se hizo la pregunta.
si la pregunta está relacionada con noticias, actualidad o si el usuario te saluda, responde con "no puedo encontrar información".
Si la respuesta no está en el contexto u obtienes una respuesta irrelevante, responde con "no puedo encontrar información".

Historial de conversación:
{chat_history}

Contexto relevante:
{context}

Pregunta actual:
{question}

Respuesta:"""

# Crear el PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=rag_template
)
emmbedings=get_hugging_face_model_embedding()

format_chain = LLMChain(llm=get_large_language_model(), prompt=prompt_template)

docs_default, processed_docs, all_docs= process_docs()

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history",
    input_key="question"
)

def search(search_input, threshold=0.7):
    try:
        

        vectorstore=vector_store(emmbedings, get_langchain_docs(docs_default))

        pdf_docs = vectorstore.similarity_search_with_score(search_input, k=3)
        
        # Filtrar documentos que tengan una similitud menor al threshold
        filtered_docs = [doc[0].page_content for doc in pdf_docs if doc[1] >= threshold]
        
        #if not filtered_docs:
            #print("RAG Agent: No se encontraron documentos relevantes.")
        return filtered_docs
    except Exception as e:
        print(f"Error en search(): {e}")
        return []
    


def rag_agent(question, threshold=0.7):  

    try:
        context = search(question, threshold)
        
        if not context:
            return {"response": "Pasar al siguiente agente"}

        context_str = "\n".join(context)
        
        # Cargar historial de memoria
        history = memory.load_memory_variables({})
        chat_history = history.get("chat_history", "")

        response = format_chain.invoke({
            "chat_history": chat_history,
            "context": context_str,
            "question": question
        })

        answer = response["text"].strip()

        for term in ["Pregunta:", "Pregunta actual:", "Explicación:", "Question:", "Explanation:", "Pregunta anterior:", "Previous question:"]:
            if term in answer:
                answer = answer.split(term)[0].strip()


        invalid_responses = ["no puedo encontrar información", "no está relacionada", "no sé"]
        if any(phrase in answer.lower() for phrase in invalid_responses) or len(answer) < 10:
            return {"response": "Pasar al siguiente agente"}
        
        # Guardar pregunta y respuesta en la memoria
        memory.save_context({"question": question}, {"response": answer})

        return {"response": answer}
    
    except Exception as e:
        print(f"Error en la generación de respuesta: {e}")
        return {"response": "Pasar al siguiente agente"}


def rag_agent_adapter(state):
    """Adapta el formato de entrada/salida del RAG Agent para funcionar con LangGraph."""
    clear_gpu_memory()  
    
    question = state.get("question", "")
    #print(f"RAG Agent recibiendo: {question}")  # Debug
    
    result = rag_agent(question)
    #print(f"RAG Agent resultado: {result}")  # Debug
    
    new_state = state.copy()
    new_state["current_agent"] = "RAG"  
    
    if isinstance(result, dict) and "response" in result:
        if result["response"] != "Pasar al siguiente agente":
            new_state["response"] = result["response"]

    return new_state
