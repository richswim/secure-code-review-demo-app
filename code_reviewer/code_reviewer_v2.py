# import os
# from dotenv import load_dotenv
# import shutil
#
# from langchain_community.document_loaders import (
#     PyPDFLoader,
#     DirectoryLoader,
#     UnstructuredMarkdownLoader,
# )
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.llms import OpenAI
# from langchain.chains import RetrievalQA
#
# load_dotenv()
#
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# BASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
#
#
# # Load PDFs
# pdf_loader = PyPDFLoader(
#     "/Users/ricardo/DEV/secure-code-review-demo-app/data/OWASP_10/OWASP_Top_10_2017.pdf"
# )
# pdf_documents = pdf_loader.load()
#
# # Load Python codebase
# code_loader = DirectoryLoader(
#     "/Users/ricardo/DEV/secure-code-review-demo-app/data/code_to_review/",
#     glob="**/*.py",
# )
# code_documents = code_loader.load()
#
# # Load Markdown documents
# markdown_loader = DirectoryLoader(
#     "/Users/ricardo/DEV/secure-code-review-demo-app/data/OWASP_10/",
#     glob="**/*.md",
# )
# markdown_documents = markdown_loader.load()
#
# # Combine documents
# documents = code_documents + markdown_documents + pdf_documents
#
# # Split text into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)
#
# # Create embeddings
# embeddings = OpenAIEmbeddings()
#
# # Create vector store
# db = Chroma.from_documents(texts, embeddings)
#
# # Initialize question-answering chain
# qa = RetrievalQA.from_chain_type(
#     llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever()
# )
#
# # Ask a question
# query = "Summarize A01:2021 – Broken Access Control"
# result = qa.run(query)
#
# print(result)
#
# """
#             <s> [INST] You are an expert programmer and OWASP 10 expert for question-answering tasks.
#             Use the following pieces of retrieved context to answer the question and suggest fixes for
#             security vulnerabilities based on the Bandit results. If you don't know the answer,
#             just say that you don't know. Use 10 sentences maximum and keep
#             the answer concise. [/INST] </s>
#
#             [INST] Question: {question}
#
#             Context:
#             {context}
#
#             Metadata:
#             {metadata}
#
#             Answer: [/INST]
#             """
import os

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

# Load the codebase from a directory
loader = DirectoryLoader(
    "/Users/ricardo/DEV/secure-code-review-demo-app/data/code_to_review/",
    glob="**/*.py",
)
documents = loader.load()

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings for the code chunks
embeddings = OpenAIEmbeddings()

# Store the embeddings in a vector store
db = FAISS.from_documents(texts, embeddings)

# Define the prompt template for code analysis
prompt_template = """
You are an AI assistant that analyzes code and provides suggestions for improvement.
Given the following code snippet:

{context}

Please provide a detailed analysis of the code, including:
1. Code quality and best practices
2. Potential bugs or vulnerabilities
3. Suggestions for optimizations and improvements
4. Any other relevant insights or recommendations
"""

prompt = PromptTemplate(
    input_variables=["context"],
    template=prompt_template,
)

# Create the retrieval-augmented QA chain
chain_type_kwargs = {"prompt": prompt}
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=db.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
)

# Analyze the codebase and provide suggestions
query = "Analyze the codebase and provide suggestions for improvement."
result = qa.run(query)

print(result)
