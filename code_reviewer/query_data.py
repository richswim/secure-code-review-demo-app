import os
from dotenv import load_dotenv

from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

BASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")


CHROMA_PATH = os.path.join(BASE_PATH, "chroma")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text

    question = ""

    while (
        question != "exit"
        or question != "quit"
        or question != "stop"
        or question != "bye"
        or question != "q"
    ):
        question = input("How can I help you? -> ")

        if question in ["exit", "quit", "stop", "bye", "q"]:
            break

        print(f"using chroma in {CHROMA_PATH}")

        # Prepare the DB.
        embedding_function = OpenAIEmbeddings()
        db = Chroma(
            persist_directory=CHROMA_PATH,
            collection_name="OWASP_10",
            embedding_function=embedding_function,
        )

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(question, k=3)

        print(results)

        if len(results) == 0 or results[0][1] < 0.7:
            print("Unable to find matching results.")
            continue

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=question)
        print(prompt)

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)


if __name__ == "__main__":
    main()
