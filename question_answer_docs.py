#!/usr/bin/env python3
import os
import time
import json

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMChainFilter,
    EmbeddingsFilter,
)
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.llms import CTransformers, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain import LLMChain
from langchain.schema.messages import messages_to_dict, messages_from_dict

from typing import Any, Dict, List
from IPython.display import Markdown, HTML, display

import openai


from constants import *

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get("PERSIST_DIRECTORY")
model_path = os.environ.get("MODEL_PATH")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 10))


# TODO: cite where the soln has been taken from
class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(
            inputs, {"response": outputs["answer"]}
        )


def retrieve_chat_history(chat_len=0):
    try:
        with open("chat_history.json", "r") as f:
            if os.stat("chat_history.json").st_size == 0:
                retrieve_from_db = []
            else:
                retrieve_from_db = json.load(f)
            # use only last 15 messages
        retrieve_from_db = retrieve_from_db[-1 * chat_len :]
    except FileNotFoundError:
        retrieve_from_db = []
    retrieved_messages = messages_from_dict(retrieve_from_db)
    return retrieved_messages


def load_chat_history(qa=True):
    """
    Load the memory from a json file.
    taken from: https://stackoverflow.com/questions/75965605/how-to-persist-langchain-conversation-memory-save-and-load
    """
    # safely load the chat_history.json file create new one if it doesn't exist
    retrieved_messages = retrieve_chat_history(CHAT_HISTORY_LEN)
    retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
    if qa:
        retrieved_memory = AnswerConversationBufferMemory(
            chat_memory=retrieved_chat_history,
            memory_key="chat_history",
            return_messages=True,
        )
    else:
        retrieved_memory = ConversationBufferMemory(
            chat_memory=retrieved_chat_history,
            memory_key="history",
            # input_key="question",
            return_messages=True,
        )
    return retrieved_memory


def save_chat_history(chain):
    """
    Load the memory from a json file.
    taken from: https://stackoverflow.com/questions/75965605/how-to-persist-langchain-conversation-memory-save-and-load
    """
    extracted_msgs = chain.memory.chat_memory.messages
    ingest_to_db = messages_to_dict(extracted_msgs)
    # safely append to the chat_history.json file
    with open("chat_history.json", "w") as f:
        json.dump(ingest_to_db, f)


def get_retriever(oss):
    if oss:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    else:
        embeddings = OpenAIEmbeddings()
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        # client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    return retriever


def get_qa_chain(llm, retriever, memory):
    """
    Get the QA chain.
    """
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        # condense_question_llm=cond_llm,
        chain_type=CHAIN_TYPE,
        return_source_documents=True,
    )
    qa.combine_docs_chain.llm_chain.prompt = QA_PROMPT
    return qa


def main(oss=False):
    # Prepare the retriever
    retriever = get_retriever(oss)
    memory = load_chat_history(qa=True)
    # Prepare the QA chain
    qa = get_qa_chain(llm, retriever, memory)
    # Interactive questions and answers over your docs
    while True:
        query = input("\nEnter a question: ")
        if query == "exit":
            # save the chat history
            save_chat_history(qa)
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        try:
            print("Thinking... Please note that this can take a few minutes.")
            start = time.time()
            res = qa({"question": query})
            answer, docs = res["answer"], res["source_documents"]
            end = time.time()

            # # Print the result
            # print("\n\n> Question:")
            # print(query)
            print(f"\n> Answer (took {round(end - start, 2)} s.):")
            # print(display(Markdown(answer)))

            # Print the relevant sources used for the answer
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
            #     print(document.page_content)
        except Exception as e:
            print(str(e))
            raise


def load_model(oss=False):
    try:
        # check if the model is already downloaded
        print("Loading model...")
        global llm, cond_llm
        # initialize llm
        if oss:
            llm = CTransformers(
                model=os.path.abspath(model_path),
                model_type="mpt",
                callbacks=[StreamingStdOutCallbackHandler()],
                config={"temperature": 0.1},
            )
        else:
            llm = ChatOpenAI(
                model_name=CHAT_MODEL,
                temperature=0.1,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
            )
            # cond_llm = OpenAI(model=COND_MODEL, temperature=0.1)
        return True

    except Exception as e:
        print(str(e))
        raise


if __name__ == "__main__":
    # load model if it has already been downloaded. If not prompt the user to download it.
    oss = bool(input("Do you want to use open source stuff? (y/n): ") == "y")
    load_model(oss)
    main(oss)
