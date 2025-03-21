import os
import sys
import textwrap
import re
import json
import pandas as pd
import nltk
import squarify
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter
from typing import Any, List, Mapping, Optional
from nltk.corpus import stopwords
from nltk.util import ngrams
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from googleapiclient.discovery import build

from huggingface_hub import notebook_login, InferenceClient
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain.llms.base import LLM
from langchain_core.documents import Document
from langchain.docstore.document import Document as DocstoreDocument
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

from model.docs_preprocessing import convert_files_to_docs, process_documents





dir_path=os.getcwd()+"/model/docs/dataset/"
print(dir_path)
#dir_path = "C:/Users/joeya/OneDrive2/OneDrive/Documentos/Cursos/Final Project/insurance_dataset/"
all_docs = convert_files_to_docs(dir_path)
docs_default, processed_docs = process_documents(all_docs)





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



nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))




def remove_stopwords(text, stop_words):
    # Dividir el texto en palabras
    words = text.split()
    # Filtrar las stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Unir las palabras filtradas en un solo texto
    return " ".join(filtered_words)


text = " ".join(doc["content"] for doc in docs_default)

# Filtrar las stopwords
filtered_text = remove_stopwords(text, stop_words)



# Limpieza adicional del texto (eliminar puntuación y convertir a minúsculas)
filtered_text = re.sub(r'[^\w\s]', '', filtered_text.lower())

# Contar la frecuencia de las palabras
word_counts = Counter(filtered_text.split())

# Seleccionar las N palabras más comunes
top_n = 20
top_words = word_counts.most_common(top_n)

# Preparar datos para el treemap
words, counts = zip(*top_words)
sizes = list(counts)
labels = [f"{word}\n({count})" for word, count in top_words]

# Crear el treemap








vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
X = vectorizer.fit_transform([doc["content"] for doc in docs_default])

df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df_tfidf.head()




# to download the model run python -m spacy download es_core_news_sm in terminal

nlp = spacy.load("es_core_news_sm") 
doc = nlp(all_docs[0]["content"]) 

for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")




# Cargar modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

# Procesar el texto
doc = nlp(text)

# Contar entidades por tipo
entities = [ent.label_ for ent in doc.ents]
entity_counts = Counter(entities)






nltk.download('punkt')
tokens = nltk.word_tokenize(docs_default[0]["content"])

bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)

print("Bigramas más comunes:", bigram_freq.most_common(10))
print("Trigramas más comunes:", trigram_freq.most_common(10))




# Convertir diccionarios a objetos Document de LangChain
langchain_docs = [
    Document(page_content=doc["content"], metadata=doc.get("meta", {}))  
    for doc in docs_default
]



embeddings_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2',
)




vectorstore = FAISS.from_documents(langchain_docs, embeddings_model)




def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




class HuggingFaceInferenceLLM(LLM):
    model_name: str = "HuggingFaceH4/zephyr-7b-alpha"
    temperature: float = 0
    max_new_tokens: int = 512
    client: Any = None  
    
    def __init__(self, model_name=None, temperature=None, max_new_tokens=None, **kwargs):
        if model_name:
            kwargs["model_name"] = model_name
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens:
            kwargs["max_new_tokens"] = max_new_tokens
        
        
        super().__init__(**kwargs)
        
        self.client = InferenceClient() #Si se loguean por terminal no es necesario poner el token acá
        #self.client = InferenceClient(token="") #Si no se loguean por terminal, poner el token acá
       
    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.text_generation(
            model=self.model_name,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature >= 0
        )
        
        if response.startswith(prompt):
            response = response[len(prompt):]
        return response
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens
        }
    


memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history",
    input_key="question"
)


llm = HuggingFaceInferenceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    memory=memory,
    temperature=0.01,
    max_new_tokens=512
)




#qa_file = "C:/Users/joeya/OneDrive2/OneDrive/Documentos/Cursos/Final Project/qna_polizas.json"
qa_file =os.getcwd()+"/model/qna/qna_polizas.json"
with open(qa_file, "r", encoding="utf-8") as file:
    qna_data = json.load(file)






faq_template = """Eres un asistente especializado en pólizas de seguros de salud.
Responde con precisión según la información dada. Responde SIEMPRE en el idioma en el que se hizo la pregunta. Usa el historial de la conversación para generar respuestas más personalizadas.
Responde de manera clara y concisa, sin agregar demasiada información irrelevante. 
Si la respuesta no está en el contexto u obtienes una respuesta irrelevante, o si la pregunta está relacionada con noticias, actualidad o si el usuario te saluda, 
responde con "No puedo encontrar información sobre esto en mis FAQs".


Contexto: {context}

Pregunta: {question}
Respuesta:"""

qa_prompt = PromptTemplate(template=faq_template, input_variables=["context", "question"])
faq_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)




qa_docs = [Document(page_content=qa["answer"], metadata={"question": qa["question"]}) for qa in qna_data]
qna_db = FAISS.from_documents(qa_docs, embeddings_model)
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

format_chain = LLMChain(llm=llm, prompt=prompt_template)




def search(search_input, threshold=0.7):
    try:
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

google_template = """Eres un asistente especializado en pólizas de seguros de salud.
Responde con precisión realizando una búsqueda en Google, en las páginas que se te proporcionó y el historial de conversación, también
puedes usar información de tu entrenamiento. 
En caso de que te saluden o haya una conversación casual que no tenga nada que ver con la información que se te proporcionó, conviertete en un asistente general y responde amablemente según tu criterio. 
Responde en el idioma en el que se hizo la pregunta. Da una respuesta clara y concisa, no incluyas información irrelevante.
Si no sabes la respuesta o la respuesta no está en el contexto responde con "no puedo encontrar información". si te piden resumen de noticias responde con "no puedo encontrar información"

Historial de conversación: {chat_history}

Contexto: {context}

Pregunta: {question}
Respuesta:"""

google_prompt = PromptTemplate(template=google_template, input_variables=["chat_history","context", "question"])





# Configuración de la API de Google
API_KEY = "AIzaSyBR7ieH9gfokxp5jCEjVKA3jgR5ZWnOIBk"  # Clave de API
SEARCH_ENGINE_ID = "c3b71a5283d724339"  #  ID de motor de búsqueda

# %%
def google_agent(query, num_results=5):
    """Realiza una búsqueda en Google y extrae información relevante."""
    try:
        # Load conversation history
        history = memory.load_memory_variables({})
        chat_history = history.get("chat_history", "")

        service = build("customsearch", "v1", developerKey=API_KEY)
        
        result = service.cse().list(
            q=query,
            cx=SEARCH_ENGINE_ID,
            num=num_results
        ).execute()

        search_results = result.get("items", [])
        if not search_results:
            return "Pasar al siguiente agente"

        context = " ".join([item.get("snippet", "") for item in search_results if item.get("snippet")])

        if not context.strip():
            return "Pasar al siguiente agente"

        document = Document(page_content=context)

        # Use the QA chain properly with all required parameters
        google_chain = load_qa_chain(llm, chain_type="stuff", prompt=google_prompt)
        full_response = google_chain.invoke({
            "input_documents": [document], 
            "question": query,
            "chat_history": chat_history
        })
        
        # Extract the response
        if isinstance(full_response, dict) and "output_text" in full_response:
            respuesta = full_response["output_text"]
        elif isinstance(full_response, str):
            respuesta = full_response.split("Respuesta:", 1)[-1].strip()
        else:
            respuesta = str(full_response)

        for term in ["Pregunta:", "Pregunta actual:", "Explicación:", "Question:", "Explanation:", "Pregunta anterior:", "Previous question:"]:
            if term in respuesta:
                respuesta = respuesta.split(term)[0].strip()


        # Si la respuesta está vacía o no aporta valor, pasar al siguiente agente
        if not respuesta or "no encontró la respuesta" in respuesta.lower():
            return "Pasar al siguiente agente"

        # Save to memory
        memory.save_context({"question": query}, {"response": respuesta})
        return respuesta

    except Exception as e:
        print(f"Error en Google Agent: {e}")
        return "Pasar al siguiente agente"

def google_agent_adapter(state):
    """Adapta la entrada/salida de Google Agent para LangGraph."""
    clear_gpu_memory()  

    question = state.get("question", "")
    #print(f"Google Agent recibiendo: {question}")  # Debug
    
    result = google_agent(question)
    #print(f"Google Agent resultado: {result}")  # Debug
    
    new_state = state.copy()
    new_state["current_agent"] = "Google"

    if (
        not result  # Si es None o vacío
        or isinstance(result, str) and len(result.strip()) < 10  
        or "no puedo encontrar" in result.lower()  
        or "no tengo información" in result.lower()
        or "lo siento" in result.lower()
    ):
        #print("Google Agent no encontró respuesta relevante, pasando al siguiente agente.")
        new_state["response"] = "Pasar al siguiente agente"
    else:
        new_state["response"] = result
        #print(f"Google Agent encontró respuesta: {result}")
        #print(f"{result}")
    
    return new_state





news_template = """Eres un asistente especializado en pólizas de seguros de salud.
Responde con precisión realizando una búsqueda sobre seguros de salud en Chile e internacionalmente. 
Genera un breve resumen de las noticias más relevantes encontradas, en lugar de listar titulares. Responde en el idioma en el que se hizo la pregunta.
Si no encuentras información relevante, dilo claramente.

Contexto: {context}

Pregunta: {question}
Resumen:"""

news_prompt = PromptTemplate(template=news_template, input_variables=["context", "question"])

API_KEY = "AIzaSyBcNMlMDTSP4L4kTgkoU6VJoVW3lfz7tYo"  # Clave de API
NEWS_ENGINE_ID = "809f46b767fe14d69" 



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



memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)




def filter_agent(state):
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
    response = re.split(r'\n\s*Consulta:', response, maxsplit=1)[0].strip()

    new_state = state.copy()
    new_state["current_agent"] = "LLM_Response"
    new_state["response"] = response

    memory.save_context({"question": query}, {"response": response})

    return {"response": response}





query_test = "Hola, soy Joey. ¿Cómo estás?"

# Crear el estado inicial
test_state = {"question": query_test}

# Probar el Filter Agent
classification = filter_agent(test_state)
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
filter_node = graph.add_node("Filter", RunnableLambda(lambda state: filter_agent(state))) #RunnableLambda(filter_agent))
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


def clear_memory():
    """Limpia la memoria de conversación."""
    memory.clear()
