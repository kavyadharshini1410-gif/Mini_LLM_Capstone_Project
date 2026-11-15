
from langchain_ollama import ChatOllama

from rag_pipeline import retrieve_documents


# Initialize the LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0.2
)


def generate_answer(question):

    print("\nRetrieving relevant information...")

    # Retrieve relevant PDF content
    documents = retrieve_documents(question)

    if not documents:
        return "I could not find relevant information in the study notes."

    # Combine retrieved content
    context = "\n\n".join(
        [
            f"Page {doc.metadata.get('page')}:\n{doc.page_content}"
            for doc in documents
        ]
    )

    # Create prompt
    prompt = f"""
You are an Education AI Study Assistant.

Answer the student's question using only the
information provided in the context below.

Give a clear and simple explanation.

If the answer is not available in the context,
say that the information is not available
in the study notes.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    print("\nGenerating AI answer...")

    # Generate answer
    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    question = "What is Machine Learning?"

    print("\nQUESTION:", question)

    answer = generate_answer(question)

    print("\nAI GENERATED ANSWER:")
    print(answer)

    print("\nChatbot response generated successfully!")
