
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "machine_learning_documents"


def get_vector_store():

    print("Connecting to ChromaDB...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

    print("Connected to ChromaDB successfully!")

    return vector_store


def retrieve_documents(question, k=3):

    vector_store = get_vector_store()

    print("\nSearching for relevant information...")

    results = vector_store.similarity_search(
        question,
        k=k
    )

    return results


if __name__ == "__main__":

    question = "What is Machine Learning?"

    print("\nQuestion:", question)

    documents = retrieve_documents(question)

    print("\nRETRIEVED DOCUMENTS:")

    for index, document in enumerate(documents, start=1):

        print(f"\nDocument {index}")
        print("Source:", document.metadata.get("source"))
        print("Page:", document.metadata.get("page"))
        print("Content:")
        print(document.page_content[:500])

    print("\nRetrieval completed successfully!")
