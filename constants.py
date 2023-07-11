import os

from chromadb.config import Settings
from dotenv import load_dotenv

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
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

template = """
You are a highly intelligent and helpful research assistant specializing in AI and its applications. 
You always answer truthfully and do not make up answers.
Your job is to research relevent information from context. And answer the questions like an expert.
If you do not find the an appropriate answer in context, just say I don't know.

{context}
"""
system_message_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        output_parser=None,
        partial_variables={},
        template=template,
        template_format="f-string",
        validate_template=True,
    ),
    additional_kwargs={},
)
human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        output_parser=None,
        partial_variables={},
        template="{question}",
        template_format="f-string",
        validate_template=True,
    ),
    additional_kwargs={},
)
PROMPT = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

CHAIN_TYPE = "stuff"
CHAT_MODEL = "gpt-3.5-turbo-0613"
COND_MODEL = "text-davinci-002"
