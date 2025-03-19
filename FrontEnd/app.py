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
    
@cl.on_chat_start
async def on_chat_start():
    
    # await cl.Message(content="Welcome to the chat!").send()

    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


async def generate_chunks(text, chunk_size=2000):
    """Yield chunks of text"""
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]

# @cl.on_message
# async def on_message(message: cl.Message):
#     runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

#     msg = cl.Message(content="")

#     async for chunk in runnable.astream(
#         {"question": message.content},
#         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
#     ):
#         await msg.stream_token(chunk)

#     await msg.send()

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="/public/idea.svg",
            ),

        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            icon="/public/learn.svg",
            ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
            icon="/public/terminal.svg",
            ),
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
            icon="/public/write.svg",
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
  

    # print("printing result " + output[0]["generated_text"][0])
    # await cl.Message(content=output[0]["generated_text"]).send()

# @cl.on_message
# async def main(message: cl.Message):
#     response = requests.post(API_URL, headers=headers, json={"inputs": message.content}, stream=True)

#     msg = cl.Message(content="")
#     for chunk in response.iter_content(chunk_size=1024):  # Stream chunks
#         msg.content += chunk.decode()
#         await msg.update()  # Send updates

#     await msg.send()  # Finalize message