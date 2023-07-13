from langchain.vectorstores import Chroma, Redis

REDIS_URL = "redis://localhost:6379"
VS_INDEX_NAME = "vectorstore"


def create_vector_store(text, embeddings, persist_directory, vs_type="redis"):
    if vs_type == "chroma":
        print(f"Created Chroma Vectorstore at {persist_directory}")
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(
            documents=text,
            persist_directory=persist_directory,
            embedding=embeddings,
        )
        db.persist()
    elif vs_type == "redis":
        print("Created Redis Vectorstore")
        print(f"Creating embeddings. May take some minutes...")
        db = Redis.from_documents(
            documents=text,
            redis_url=REDIS_URL,
            index_name=VS_INDEX_NAME,
            embedding=embeddings,
        )
    else:
        raise ValueError("Vectorstore type not supported")
    return db


def load_vector_store(
    embeddings, persist_directory, index_name=VS_INDEX_NAME, vs_type="redis"
):
    if vs_type == "chroma":
        print(f"Appending to existing Chroma Vectorstore at {persist_directory}")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
    elif vs_type == "redis":
        print("Updating existing Redis Vectorstore")
        db = Redis.from_existing_index(
            redis_url=REDIS_URL,
            index_name=index_name,
            embedding=embeddings,
        )
    else:
        raise ValueError("Vectorstore type not supported")
    return db
