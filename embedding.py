
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from preprocessing import load_and_split_pdf


# Project directory
BASE_DIR = Path(__file__).parent

# ChromaDB storage location
CHROMA_DIR = BASE_DIR / "chroma_db"

# Collection name
COLLECTION_NAME = "machine_learning_documents"


def create_vector_store():

    print("Loading preprocessed PDF chunks...")

    # Load PDF chunks
    chunks = load_and_split_pdf()

    print("\nInitializing embedding model...")

    # Initialize Ollama embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    print("\nGenerating embeddings and storing in ChromaDB...")

    # Create and store vector database
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR)
    )

    # Verify stored vectors
    total_vectors = vector_store._collection.count()

    print("\nEmbedding process completed successfully!")
    print("Total chunks processed:", len(chunks))
    print("Total vectors stored:", total_vectors)
    print("Database location:", CHROMA_DIR)

    return vector_store


if __name__ == "__main__":

    create_vector_store()
