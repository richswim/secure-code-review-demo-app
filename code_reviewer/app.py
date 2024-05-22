import os
import sys

import streamlit as st
from streamlit_chat import message

BASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

sys.path.append(BASE_PATH)
from code_reviewer.chat_secure import ChatCode

st.set_page_config(page_title="ChatCode")


def display_messages():
    """
    Displays chat messages in the app.

    Args:

    Returns:
        None
    """

    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    """
    Processes user input, interacts with the assistant,
    and updates chat messages accordingly.

    Args:

    Returns:
        None
    """

    if (
        st.session_state["user_input"]
        and len(st.session_state["user_input"].strip()) > 0
    ):
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)
            st.session_state["user_input"] = ""

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    """
    Processes user input, interacts with the assistant,
    and updates chat messages accordingly.

    Args:

    Returns:
        None
    """

    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    file_path = st.session_state["txt_input"]
    st.session_state["assistant"].ingest(file_path)
    st.session_state["messages"].append(
        (
            f"Code ingested from {file_path}. Ready to chat!",
            False,
        )
    )


def page():
    """
    Displays the main page of the app, initializes session state if needed,
    and sets up the chat interface.

    Args:

    Returns:
        None
    """

    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatCode()

    st.header("Code Review Chatbot")

    st.text_input(
        "/Users/ricardo/DEV/secure-code-review-demo-app/data/code_to_review/",
        placeholder="some directory",
        on_change=read_and_save_file,
        key="txt_input",
    )

    st.session_state["ingestion_spinner"] = st.empty()
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
