import os
from dataclasses import asdict, dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import openai

from utils import format_prompt
from constants import CHAT_MODEL, CHAT_PROMPT
from question_answer_docs import load_chat_history, save_chat_history

load_dotenv()

model_path = os.environ.get("MODEL_PATH")
openai.api_key = os.environ.get("OPENAI_API_KEY")

# @dataclass
# class GenerationConfig:
#     # sample
#     top_k: int
#     top_p: float
#     temperature: float
#     repetition_penalty: float
#     # last_n_tokens: int
#     seed: int

#     # eval
#     # batch_size: int
#     # threads: int

#     # generate
#     max_new_tokens: int
#     # stop: list[str]
#     stream: bool
#     # reset: bool


def get_conv_chain():
    # load the memory from a json file
    retrieved_memory = load_chat_history(qa=False)
    return ConversationChain(llm=llm, memory=retrieved_memory, prompt=CHAT_PROMPT)


def load_model(oss=False):
    try:
        # check if the model is already downloaded
        print("Loading model...")
        global llm
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
        return True

    except Exception as e:
        print(str(e))
        raise


if __name__ == "__main__":
    # load model if it has already been downloaded. If not prompt the user to download it.
    oss = bool(input("Do you want to use open source stuff? (y/n): ") == "y")
    load_model(oss)
    chain = get_conv_chain()
    # generation_config = GenerationConfig(
    #     temperature=0.1,
    #     top_k=0,
    #     top_p=0.9,
    #     repetition_penalty=1.0,
    #     max_new_tokens=512,
    #     seed=42,
    #     # reset=False,
    #     stream=True,  # streaming per word/token
    #     # threads=int(os.cpu_count() / 2),  # adjust for your CPU
    #     # stop=["<|im_end|>", "|<"],
    #     # last_n_tokens=64,
    #     # batch_size=8,
    # )

    while True:
        query = input("\nEnter a question: ")
        if query == "exit":
            # save the chat history
            save_chat_history(chain)
            break
        if query.strip() == "":
            continue
        try:
            print("Thinking...")
            # call llm with formatted user prompt and generation config
            response = chain({"input": query})
            # print response
            print("\n")
        except Exception as e:
            print(str(e))
            raise
