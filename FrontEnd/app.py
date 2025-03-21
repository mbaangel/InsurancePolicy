from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
import openai
import requests
import os

import chainlit as cl

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b"
headers = {"Authorization": "Bearer " + HUGGINGFACE_API_KEY}

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
    



async def generate_chunks(text, chunk_size=2000):
    """Yield chunks of text"""
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]



@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Denuncia siniestro",
            message="¿Cómo se denuncia un siniestro o accidente?",
            icon="../public/accident.svg",
            ),

        cl.Starter(
            label="Calculo gasto reembolsable",
            message="¿Cómo se calcula un gasto reembolsable?",
            icon="../public/rembolso.svg",
            ),
        cl.Starter(
            label="Accidente según la póliza POL120190177",
            message="¿Qué se considera un Accidente según la póliza POL120190177?",
            icon="../public/poliza.svg",
            ),
        cl.Starter(
            label="No pago de prima",
            message="¿Qué sucede si el asegurado no paga la prima?",
            icon="../public/prima.svg",
            )
        ]


@cl.on_message
async def on_message(message: cl.Message):
    response = requests.post(API_URL, headers=headers, json={"inputs": message.content})
    output = response.json()
    if isinstance(output, list) and "generated_text" in output[0]:

        msg = cl.Message(content="")
        generated_text = output[0]["generated_text"]
        async for chunk in generate_chunks(generated_text):
                msg.content += chunk  # Append chunk
                await msg.update()  # Send updated message
        await msg.send()
    else:   
        await cl.Message(content="Error: Invalid API response").send()
  

