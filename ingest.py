import glob
import os
from multiprocessing import Pool
from typing import List

import openai
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt

from constants import CHROMA_SETTINGS

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
#  Load environment variables
persist_directory = os.environ.get("PERSIST_DIRECTORY")
# directory where source documents to be ingested are located
source_directory = os.environ.get("SOURCE_DIRECTORY", "source_documents")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
chunk_size = 1200
chunk_overlap = 200
oss = False

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    # ".csv": (CSVLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {"encoding": "latin"}),
    ".md": (UnstructuredMarkdownLoader, {}),
    # ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        # print(f"Loading {file_path}")
        return loader.load()
    else:
        pass

    # raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    print(f"Found {len(all_files)} files")
    print(f"Ignoring {len(ignored_files)} files")
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]
    print(f"Loading {len(filtered_files)} new documents")

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for i, docs in enumerate(
                pool.imap_unordered(load_single_document, filtered_files)
            ):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    get_emb_cost_estimates(documents)
    proceed = input("Do you want to proceed? (y/n): ")
    if proceed == "n":
        exit(0)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, "index")):
        if os.path.exists(
            os.path.join(persist_directory, "chroma-collections.parquet")
        ) and os.path.exists(
            os.path.join(persist_directory, "chroma-embeddings.parquet")
        ):
            list_index_files = glob.glob(os.path.join(persist_directory, "index/*.bin"))
            list_index_files += glob.glob(
                os.path.join(persist_directory, "index/*.pkl")
            )
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


def get_emb_cost_estimates(docs):
    total_characters = 0
    for doc in docs:
        total_characters += len(doc.page_content)

    total_tokens = total_characters / 4
    emb_cost = total_tokens * 0.0001 / 1000  # 0.0001 USD per 1000 tokens

    print("Total number of characters:", total_characters)
    print("Total number of tokens:", total_tokens)
    print("Cost of creating embeddings:", emb_cost)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def create_new_db(texts, embeddings):
    """
    function to create new a new vectorstore with exponential backoff to avoid rate limit error
    """
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_directory,
        client_settings=CHROMA_SETTINGS,
    )
    return db


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def add2db(db, texts):
    """
    function to add more documents to an existing vectorstore
    """
    db.add_documents(texts)
    return db


def main():
    # Create embeddings
    oss = bool(input("Do you want to use open source stuff? (y/n): ") == "y")
    if oss:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    else:
        embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db_exists = does_vectorstore_exist(persist_directory)
    if db_exists:
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
        collection = db.get()
        texts = process_documents(
            [metadata["source"] for metadata in collection["metadatas"]]
        )
        print(f"Creating embeddings. May take some minutes...")
        db = add2db(db, texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = create_new_db(texts, embeddings)
    db.persist()
    db = None

    print(
        f"Ingestion complete! You can now run chat_with_docs.py to query your documents"
    )


if __name__ == "__main__":
    main()
