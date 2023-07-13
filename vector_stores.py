from langchain.vectorstores import Chroma, Redis

from embeddings import *

REDIS_URL = "redis://localhost:6379"
VS_INDEX_NAME = "vectorstore"

emb_type = str(
    input("choose embedding type (openai/hf): ")
)  # not sure if this is the best way to do this
embeddings = get_embeddings(emb_type)


def create_chroma_vector_store(text, persist_directory):
    print(f"Created Chroma Vectorstore at {persist_directory}")
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(
        documents=text,
        persist_directory=persist_directory,
        embedding=embeddings,
    )
    db.persist()
    return db


def create_redis_vector_store(text, metadata):
    print("Created Redis Vectorstore")
    print(f"Creating embeddings. May take some minutes...")
    db, keys = Redis.from_texts_return_keys(
        texts=text,
        metadatas=metadata,
        redis_url=REDIS_URL,
        index_name=VS_INDEX_NAME,
        embedding=embeddings,
    )
    return db, keys


def load_redis_vector_store(index_name=VS_INDEX_NAME):
    print("Updating existing Redis Vectorstore")
    db = Redis.from_existing_index(
        redis_url=REDIS_URL,
        index_name=index_name,
        embedding=embeddings,
    )
    return db


def load_chroma_vector_store(persist_directory):
    print(f"Appending to existing Chroma Vectorstore at {persist_directory}")
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return db
