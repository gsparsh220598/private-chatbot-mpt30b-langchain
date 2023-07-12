import os

from chromadb.config import Settings
from dotenv import load_dotenv

from langchain import PromptTemplate

load_dotenv()

# Define the folder for storing database on disk and load
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY")

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    # Optional, defaults to .chromadb/ in the current directory
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)

condense_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

qa_template = """
You are a highly intelligent and helpful research assistant specializing in AI and its applications. 
You always answer truthfully and do not make up answers.
Your job is to research relevent information from context and the chat history. And answer the questions like an expert.
If you do not find the an appropriate answer in context, just say I don't know.

chat history:
{chat_history}

Context:
{context}

Question:
{question}
Helpful Answer in Markdown:
"""

chat_template = """
You are a helpful AI assistant. You answer truthfully and do not make up answers.
Your job is to answer the user's questions referring to the chat history when required.
If you do not know the answer, just say I don't know.

chat history:
{history}

Human: {input}
AI:
"""

QA_PROMPT = PromptTemplate(
    template=qa_template, input_variables=["chat_history", "context", "question"]
)
CONDENSE_PROMPT = PromptTemplate(
    template=condense_template, input_variables=["chat_history", "question"]
)
CHAT_PROMPT = PromptTemplate(
    template=chat_template, input_variables=["history", "input"]
)
CHAIN_TYPE = "stuff"
CHAT_MODEL = "gpt-3.5-turbo-0613"
CONDENSE_MODEL = "text-davinci-002"
CHAT_HISTORY_LEN = 15
