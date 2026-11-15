
from pathlib import Path
import re

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# PDF file location
PDF_PATH = (
    Path(__file__).parent
    / "dataset"
    / "documents"
    / "Machine_Learning_Study_Notes.pdf"
)


def clean_text(text):
    """Clean unnecessary spaces and line breaks."""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_and_split_pdf():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # Read PDF
    reader = PdfReader(str(PDF_PATH))

    documents = []

    for page_number, page in enumerate(
        reader.pages, start=1
    ):

        text = page.extract_text() or ""
        text = clean_text(text)

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": PDF_PATH.name,
                        "page": page_number
                    }
                )
            )

    if not documents:
        raise ValueError(
            "No readable text found in the PDF."
        )

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print("PDF loaded successfully!")
    print("Total pages:", len(reader.pages))
    print("Pages with text:", len(documents))
    print("Total chunks created:", len(chunks))

    print("\nFirst chunk preview:")
    print(chunks[0].page_content[:500])

    return chunks


if __name__ == "__main__":
    chunks = load_and_split_pdf()
