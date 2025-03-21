from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from googleapiclient.discovery import build
from langchain.prompts import PromptTemplate
from model.utils import clear_gpu_memory

news_template = """Eres un asistente especializado en pólizas de seguros de salud.
Responde con precisión realizando una búsqueda sobre seguros de salud en Chile e internacionalmente. 
Genera un breve resumen de las noticias más relevantes encontradas, en lugar de listar titulares. Responde en el idioma en el que se hizo la pregunta.
Si no encuentras información relevante, dilo claramente.

Contexto: {context}

Pregunta: {question}
Resumen:"""

news_prompt = PromptTemplate(template=news_template, input_variables=["context", "question"])

API_KEY = "AIzaSyBcNMlMDTSP4L4kTgkoU6VJoVW3lfz7tYo"  # Clave de API
NEWS_ENGINE_ID = "809f46b767fe14d69"  # ID de motor de búsqueda



def news_agent(query, num_results=10):
    """Busca noticias recientes sobre seguros de salud en Google News y genera un breve resumen."""
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)

        result = service.cse().list(
            q=f"seguros de salud {query}",
            cx=NEWS_ENGINE_ID,
            num=num_results,
            #searchType="news"
        ).execute()

        search_results = result.get("items", [])
        if not search_results:
            return "No se encontraron noticias relevantes sobre seguros de salud."

        context = " ".join([item.get("snippet", "") for item in search_results if item.get("snippet")])

        document = Document(page_content=context)

        gprompt = news_prompt.format(context=context, question=query)

        google_news_chain = load_qa_chain(llm, chain_type="stuff", prompt=news_prompt)

        full_response = google_news_chain.run(input_documents=[document], question=query)

        resumen = full_response.split("Resumen:", 1)[-1].strip()

        return resumen

    except Exception as e:
        print(f"Error en google_news_agent: {e}")
        return "Error al obtener noticias de Google."


def news_agent_adapter(state):
    """Adapta el formato de entrada/salida del Google News Agent para LangGraph."""
    clear_gpu_memory()

    question = state.get("question", "")
    #print(f"Google News Agent recibiendo: {question}")  # Debug

    result = news_agent(question)

    new_state = state.copy()

    if result and "No se encontraron noticias" not in result:
        new_state["response"] = result
        #print(f"Google News Agent encontró respuesta: {result}")
    else:
        new_state["response"] = "Pasar al siguiente agente"
        print("No he podido encontrar respuesta, prueba reformulando la pregunta.")

    new_state["current_agent"] = "Google News"

    return new_state