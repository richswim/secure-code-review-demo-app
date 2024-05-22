import os
from typing import Optional

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import Language, CharacterTextSplitter

from code_reviewer.bandit_executor import bandit_process

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


class ChatCode:
    """
    Class representing a chat code with functionalities to ingest data, ask questions, and retrieve metadata.

    ChatCode class has the following attributes:
    - chain: Represents the chat code pipeline.
    - owasp_retriever: Retriever object for OWASP related documents.
    - query: String representing the query to be asked.
    - retriever: Retriever object for all relevant documents.
    - memory: Memory buffer for conversation history.
    - model: ChatOllama model object.
    - prompt: Template for the conversation prompt.

    Methods in the ChatCode class:

    __init__(self): Initializes the ChatCode object and sets up the required attributes.

    ingest(self, path: str) -> None: Ingests the data from the given path and sets up the retriever and chain attributes.

    _process_bandit_results(self, path: str) -> list[Document]: Processes Bandit results from the given path and returns a list of Document objects.

    _process_owasp_md(self, path: str) -> list[Document]: Processes OWASP markdown documents from the given path and returns a list of Document objects.

    _process_python_code(self, path: str) -> list[Document]: Processes Python code files from the given path and returns a list of Document objects.

    ask(self, query: str) -> str: Returns the answer to the given query using the chain pipeline.

    get_metadata(self, input: str) -> str: Retrieves metadata from relevant documents based on the input and returns a string containing the metadata.

    clear(self) -> None: Resets all the attributes of the ChatCode object.
    """

    chain = None

    def __init__(self):
        """
        Initializes the attributes for OWASP retriever, query, retriever, memory, model, and prompt template.
        """

        self.owasp_retriever = None
        self.query: str = Optional[None]
        self.retriever = None
        self.memory = None
        self.model = ChatOllama(model="gemma:7b")
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an brilliant software security expert.
            Use the retrieved Context to answer the question, If the provided Context does not contain any information 
            regarding the question, use  your own knowledge to answer the question. 
            If you don't know the answer, 
            just say that you don't know. [/INST] </s>

            [INST] Question: {question} 

            Context:
            {context}
            
            Answer: [/INST]
            """
        )

    def ingest(self, path: str) -> None:
        """
        Ingests and processes Python code and Bandit scan results from the specified path,
        creates vector representations, and sets up the retrieval chain for further processing.

        Args:
            path: The path to the directory containing Python code and Bandit scan result files.

        Returns:
            None
        """

        chunk_python = self._process_python_code(path)

        chunks_bandit = self._process_bandit_results(path)

        chunks = chunk_python + chunks_bandit

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
        )

        self.retriever = vector_store.as_retriever(
            # search_type="similarity_score_threshold",
            search_type="mmr",
            search_kwargs={"k": 6, "score_threshold": 0.3},
        )

        # self.memory = ConversationBufferMemory(memory_key="history")

        self.chain = (
            {
                "context": self.retriever,
                # "owasp": self.owasp_retriever,  # "owasp": self.retriever,
                "question": RunnablePassthrough(),
                # "history": self.memory.load_memory_variables,
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def _process_bandit_results(self, path: str) -> list[Document]:
        """
        Processes Bandit scan results located at the specified path.

        Args:
            path: The path to the directory containing Bandit scan result files.

        Returns:
            list[Document]: A list of Document objects representing processed Bandit scan result chunks.

        Raises:
            ValueError: If no Bandit result files are found at the given path or if an error occurs during processing.
        """

        # Load Bandit results
        bandit_process(path)
        bandit_loader = DirectoryLoader(
            path="/Users/ricardo/DEV/secure-code-review-demo-app/data/bandit",
            glob="**/*.txt",
        )
        bandit_documents = bandit_loader.load()
        if not bandit_documents:
            raise ValueError(f"No Bandit Result Files found at given path: {path}")
        # Split bandit into chunks
        text_splitter_bandit = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        if chunks_bandit := [
            Document(
                page_content=chunk.page_content,
                metadata={"type_of_data": "bandit", **chunk.metadata},
            )
            for chunk in text_splitter_bandit.split_documents(bandit_documents)
        ]:
            return chunks_bandit
        else:
            raise ValueError("Something went wrong in processing Bandit documents")

    def _process_owasp_md(self, path: str) -> list[Document]:
        """
        Processes OWASP Top 10 Markdown documents located at the specified path.

        Args:
            path: The path to the directory containing OWASP Top 10 Markdown files.

        Returns:
            list[Document]: A list of Document objects representing processed OWASP Markdown document chunks.

        Raises:
            ValueError: If no OWASP Top 10 files are found at the given path or if an error occurs during processing.
        """

        # Load Markdown documents
        markdown_loader = DirectoryLoader(
            path="/Users/ricardo/DEV/secure-code-review-demo-app/data/OWASP_10/",
            glob="**/*.md",
        )
        markdown_documents = markdown_loader.load()
        if not markdown_documents:
            raise ValueError(f"No OWASP Top 10 Files found at given path: {path}")
        # Split MDs into chunks
        text_splitter_md = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        if chunks_owasp := [
            Document(
                page_content=chunk.page_content,
                metadata={"type_of_data": "owasp", **chunk.metadata},
            )
            for chunk in text_splitter_md.split_documents(markdown_documents)
        ]:
            return chunks_owasp
        else:
            raise ValueError("Something went wrong in processing OWASP documents")

    def _process_python_code(self, path: str) -> list[Document]:
        """
        Processes Python code files located at the specified path.

        Args:
            path: The path to the directory containing Python code files.

        Returns:
            list[Document]: A list of Document objects representing processed Python code chunks.

        Raises:
            ValueError: If no Python files are found at the given path or if an error occurs during processing.
        """

        # Load Python files
        code_base_loader = DirectoryLoader(
            path=path,
            glob="**/*.py",
        )
        python_files = code_base_loader.load()
        if not python_files:
            raise ValueError(f"No Python Files found at given path: {path}")
        # Split Code base into chunks
        text_splitter_python = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        if chunk_python := [
            Document(
                page_content=chunk.page_content,
                metadata={"type_of_data": "code", **chunk.metadata},
            )
            for chunk in text_splitter_python.split_documents(python_files)
        ]:
            return chunk_python
        else:
            raise ValueError("Something went wrong in processing Python documents")

    def ask(self, query: str) -> str:
        """
        Retrieves metadata from relevant documents based on the input.

        Args:
            input: The input string to retrieve metadata for.

        Returns:
            str: A string containing metadata in the format "key: value" for each document.

        Raises:
            Any specific exceptions that may be raised during the process.
        """
        if not self.chain:
            return "Please, add a directory first."

        return self.chain.invoke(query)
        # self.memory.save_context({"input": query}, {"output": result})
        # return result

    def get_metadata(self, input: str) -> str:
        """
        Retrieves metadata from relevant documents based on the input.

        Args:
            input: The input string to retrieve metadata for.

        Returns:
            str: A string containing metadata in the format "key: value" for each document.

        Raises:
            Any specific exceptions that may be raised during the process.
        """

        retrieved_docs = self.retriever.get_relevant_documents(input)
        print(retrieved_docs)
        metadata = "\n".join(
            [
                f"{key}: {value}"
                for doc in retrieved_docs
                for key, value in doc.metadata.items()
            ]
        )
        print(metadata)

        return metadata

    def clear(self) -> None:
        """
        Clears the attributes related to chain, retriever, OWASP retriever, query, and memory.

        Returns:
            None
        """

        self.chain = None
        self.retriever = None
        self.owasp_retriever = None
        self.query = None
        self.memory = None
