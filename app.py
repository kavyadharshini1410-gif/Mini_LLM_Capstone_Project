
import streamlit as st
import csv
from datetime import datetime
from pathlib import Path

from chatbot import generate_answer


# Project paths
BASE_DIR = Path(__file__).parent
CHAT_HISTORY_FILE = BASE_DIR / "dataset" / "chat_history.csv"


# Save conversation to CSV
def save_chat_history(question, answer):

    file_exists = (
        CHAT_HISTORY_FILE.exists()
        and CHAT_HISTORY_FILE.stat().st_size > 0
    )

    with open(
        CHAT_HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp", "question", "answer"]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer
        })


# Page configuration
st.set_page_config(
    page_title="Education AI Study Assistant",
    page_icon="📚",
    layout="centered"
)


# Application title
st.title("📚 Education AI Study Assistant")

st.write(
    "Ask questions about your Machine Learning study notes. "
    "The AI assistant will find relevant information "
    "from your PDF and generate simple answers."
)


# Sidebar
with st.sidebar:
    st.header("About the Project")
    st.write("Education AI Study Assistant")
    st.write("Powered by Llama 3.2 and LangChain.")
    st.write("Uses RAG and ChromaDB for document retrieval.")
    st.divider()
    st.write("Source: Machine_Learning_Study_Notes.pdf")
    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
question = st.chat_input("Ask your question here...")


if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            try:
                answer = generate_answer(question)

            except Exception as error:
                answer = f"An error occurred: {error}"

        st.markdown(answer)

    # Save conversation
    save_chat_history(question, answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
