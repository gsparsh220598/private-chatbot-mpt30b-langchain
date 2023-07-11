#!/usr/bin/env python3
import os
import time

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMChainFilter,
    EmbeddingsFilter,
)
from langchain.memory import ConversationBufferMemory
from langchain.llms import CTransformers, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma

from typing import Any, Dict
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


def main(oss=False):
    # Prepare the retriever
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
    memory = AnswerConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        # condense_question_llm=cond_llm,
        chain_type=CHAIN_TYPE,
        return_source_documents=True,
    )
    qa.combine_docs_chain.llm_chain.prompt = PROMPT
    # Interactive questions and answers over your docs
    while True:
        query = input("\nEnter a question: ")
        if query == "exit":
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

            # # Print the relevant sources used for the answer
            # for document in docs:
            #     print("\n> " + document.metadata["source"] + ":")
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
