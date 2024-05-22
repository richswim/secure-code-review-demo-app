import os
from dotenv import load_dotenv
import shutil

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from logger import logger

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

CHROMA_PATH = os.path.join(BASE_PATH, "chroma")
DOCUMENTS_PATH = os.path.join(BASE_PATH, "data/OWASP_10/docs/")
CODE_PATH = os.path.join(BASE_PATH, "data/code_to_review/")

OWASP_COLLECTION_NAME = "OWASP_10"
CODE_COLLECTION_NAME = "CODE_TO_REVIEW"

CODE_GLOB = "*.py"
DOCUMENT_GLOB = "*.md"
EXCLUDE = "!*?.*?.md"


def main():
    collections = [
        (OWASP_COLLECTION_NAME, DOCUMENT_GLOB, EXCLUDE, DOCUMENTS_PATH),
        (CODE_COLLECTION_NAME, CODE_GLOB, None, CODE_PATH),
    ]

    for collection_name, grob, exclude, path in collections:
        generate_data_store(
            collection_name=collection_name, glob=grob, exclude=exclude, path=path
        )


def generate_data_store(collection_name: str, glob: str, exclude: str, path: str):
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks=chunks, collection_name=collection_name, path=path)


def load_documents(collection_name: str, glob: str, exclude: str):
    logger.info(f"Loading documents from {DOCUMENTS_PATH}.")

    loader = DirectoryLoader(
        DOCUMENTS_PATH, glob="*.md", recursive=True, exclude="!*?.*?.md"
    )
    return loader.load()


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=10,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]

    return chunks


def save_to_chroma(chunks: list[Document], collection_name: str, path: str):
    # Clear out the database first.
    if os.path.exists(path):
        shutil.rmtree(path)

    if not collection_name:
        logger.error("No collection name provided.")
        raise ValueError("No collection name provided.")

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        persist_directory=path,
        collection_name=collection_name,
    )
    db.persist()
    logger.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
