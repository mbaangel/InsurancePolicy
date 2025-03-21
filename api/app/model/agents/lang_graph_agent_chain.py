from model.utils import clear_gpu_memory
from model.large_langueje_model import get_large_language_model
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END
from model.agents.faq_agent_adapter import faq_agent_adapter
from model.agents.rag_agent_definition import rag_agent_adapter
from model.agents.google_agent_adapter import google_agent_adapter
from model.agents.news_agent_adapter import news_agent_adapter



memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

llm= get_large_language_model()



def filter_agent(state, memory):
    """Determina si la pregunta debe ir al sistema especializado o al asistente general."""
    query = state.get("question", "")

    classifier_prompt = f"""Tu tarea es clasificar si una consulta debe ser respondida por el sistema especializado en seguros de salud o por un asistente general.

    Reglas:

    La consulta SIEMPRE debe entrar al sistema especializado salvo en los siguientes casos en los cuales se debe usar el asistente general:

    - Un saludo o despedida
    - Datos personales como nombre, dirección, etc.

    2. De lo contrario la consulta SIEMPRE debe entrar al sistema especializado.

    Si te preguntan por noticias o actualidad, SIEMPRE debe responder el sistema especializado

    Responde SOLO con "CHAIN" si la consulta debe entrar al sistema especializado, o "LLM" si debe ser respondida por el asistente general.

    3. Responde ÚNICAMENTE con la palabra "CHAIN" o "LLM"
    4. NO des explicaciones ni ejemplos
    5. NO repitas la pregunta
    6. Si la pregunta está en otro idioma, utiliza exactamente las mismas reglas

    Consulta: {query}

    Respuesta:"""
    
    classification = llm.invoke(classifier_prompt).strip()
    #classification = re.split(r'\n\s*Consulta:', response, maxsplit=1)[0].strip()
    
    classification = classification.split("\n")[0].strip()
    #print(f"🤖 Filter Agent - Clasificación: {classification}")

    new_state = state.copy()
    new_state["current_agent"] = "Filter"

    if classification == "CHAIN":
        #print("✅ Clasificado como CHAIN - Redirigiendo a FAQ")
        new_state["next_agent"] = "FAQ"
        new_state["current_agent"] = "FAQ"
        #new_state["response"] = None 
    else:
        #print("ℹ️ Clasificado como LLM - Redirigiendo a LLM_Response")
        new_state["next_agent"] = "LLM_Response"
       
    #print(f"🤖 Filtro: Estado después de clasificación: {new_state}")
    #memory.save_context({"question": query}, {"response": classification})

    return new_state

def llm_response_agent(state, memory):
    history = memory.load_memory_variables({})
    """Agente LLM predefinido para responder preguntas fuera del ámbito de seguros de salud de manera libre y relevante."""
    query = state.get("question", "")

    llm_prompt = f"""Responde de manera breve y amigable a la siguiente consulta, sin agregar información innecesaria. Responde en el mismo idioma en que te hicieron la pregunta:

    Historial de conversación: {history}
    Consulta: {query}
    Respuesta:"""

    response = llm.invoke(llm_prompt).strip()
    #response = re.split(r'\n\s*Consulta:', response, maxsplit=1)[0].strip()

    new_state = state.copy()
    new_state["current_agent"] = "LLM_Response"
    new_state["response"] = response

    memory.save_context({"question": query}, {"response": response})

    return {"response": response}



query_test = "Hola, soy Joey. ¿Cómo estás?"

# Crear el estado inicial
test_state = {"question": query_test}

# Probar el Filter Agent
classification = filter_agent(test_state,memory)
print(f"Clasificación: {classification}")

# Si el filtro lo manda al LLM, probar la respuesta del LLM
if classification == "LLM_Response":
    response = llm_response_agent(test_state)
    print(f"Respuesta del LLM: {response['response']}")




def router(state):
    """Determina el siguiente nodo basado en el estado actual."""
    #print(f"Router state antes de decisión: {state}")
    
    # Si ya tenemos una respuesta válida, terminamos
    if "response" in state and state["response"] and state["response"] != "Pasar al siguiente agente":
        #print(f"Respuesta válida encontrada en router: {state['response']}")
        return END

    # Flujo entre agentes
    if state.get("current_agent") == "FAQ":
        #print("Pasando de FAQ a RAG")
        return "RAG"

    if state.get("current_agent") == "RAG":
        #print("Pasando de RAG a Google")
        return "Google"

    if state.get("current_agent") == "Google":
        #print("Pasando de Google a News")
        return "News"

    if state.get("current_agent") == "News":
        #print("Finalizando flujo en News")
        return END

    #print("Finalizando flujo por defecto")
    return END

def filter_router(state):
    """Función de enrutamiento para el filtro"""
    next_agent = state.get("next_agent")
    if next_agent in ["FAQ", "LLM_Response"]:
        return next_agent
    return END

# Crear el grafo de estados
graph = StateGraph(state_schema=dict)

# Agregar nodos (agentes)
filter_node = graph.add_node("Filter", RunnableLambda(lambda state: filter_agent(state, memory))) #RunnableLambda(filter_agent))
llm_response_node = graph.add_node("LLM_Response", RunnableLambda(lambda state: llm_response_agent(state, memory))) #RunnableLambda(llm_response_agent))
faq_node = graph.add_node("FAQ", RunnableLambda(faq_agent_adapter))
rag_node = graph.add_node("RAG", RunnableLambda(rag_agent_adapter))
google_node = graph.add_node("Google", RunnableLambda(google_agent_adapter))
news_node = graph.add_node("News", RunnableLambda(news_agent_adapter))

graph.add_conditional_edges(
    "Filter",
    filter_router,
    {
        "FAQ": "FAQ",
        "LLM_Response": "LLM_Response",
        END: END
    }
)

graph.add_conditional_edges(
    "FAQ",
    router,
    #lambda state: print(f"Estado después de FAQ: {state}") or router(state),
    {
        "RAG": "RAG",
        END: END
    }
)

graph.add_conditional_edges(
    "RAG",
    router,
    {
        "Google": "Google",
        END: END
    }
)

graph.add_conditional_edges(
    "Google",
    router,
    {
        "News": "News",
        END: END
    }
)

graph.add_edge("News", END)
graph.add_edge("LLM_Response", END)

graph.set_entry_point("Filter")

chain = graph.compile()

def clear_memory():
    """Limpia la memoria de conversación."""
    memory.clear()


def Final_Agent(query):
    """Executes the LangGraph flow and returns the first valid response."""
    #print(f"Procesando consulta: {query}")
    
    try:
        clear_gpu_memory()

        chat_history = memory.load_memory_variables({}).get("chat_history", [])
        
        # Crear el estado inicial incluyendo el historial
        initial_state = {
            "question": query,
            "current_agent": "Filter",
            "chat_history": chat_history
        }


        result = chain.invoke(initial_state)
        #print(f"Resultado del grafo (tipo: {type(result)}): {result}")

        for term in ["Pregunta:", "Pregunta actual:", "Explicación:", "Question:", "Pregunta anterior:", "Previous question:"]:
            if term in result:
                result = result.split(term)[0].strip()


        # Verificar si el resultado es un diccionario válido
        if isinstance(result, dict):
            #print(f"Estado final después del grafo: {result}")

            # Si hay una respuesta válida, devolverla
            if "response" in result and result["response"] and result["response"] != "Pasar al siguiente agente":
                #print(f"Respuesta final encontrada: {result['response']}")
                #return f"Answer: {result['response']}"
                memory.save_context(
                    {"input": query},
                    {"output": result["response"]}
                )

                return f"{result['response']}"
        
        # Si el resultado es un string válido, devolverlo
        
        elif isinstance(result, str) and result and result != "Pasar al siguiente agente":

            memory.save_context(
                {"input": query},
                {"output": result}
            )
            
            #print(f"Respuesta string encontrada: {result}")
            #return f"Answer: {result}"
            return f"{result}"

        # Si ninguna de las condiciones anteriores se cumple, no hay respuesta válida
        #print("No se encontró una respuesta adecuada.")
        return "No se encontró una respuesta adecuada."
    
    except Exception as e:
        print(f"Error en Final_Agent: {e}")
        return f"Error procesando la consulta: {str(e)}"


