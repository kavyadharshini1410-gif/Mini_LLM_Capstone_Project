# Education AI Study Assistant – Mini LLM Project

## Project Overview
The Education AI Study Assistant is an AI-powered application that answers questions using information from Machine Learning study notes in a PDF document.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate simple answers.

## Technologies Used
- Python 3.12
- Streamlit
- LangChain
- Ollama
- Llama 3.2
- Nomic Embed Text
- ChromaDB
- Pandas

## Key Features
- PDF document preprocessing
- Text chunking
- Text embedding generation
- Vector database storage
- Retrieval-Augmented Generation (RAG)
- AI-powered question answering
- Interactive Streamlit chatbot
- Chat history storage in CSV
- Clear Chat option
- Chatbot evaluation and analysis

## Project Structure
- `app.py` – Streamlit application
- `chatbot.py` – Generates AI answers
- `preprocessing.py` – Extracts and splits PDF text
- `embedding.py` – Creates and stores embeddings
- `rag_pipeline.py` – Retrieves relevant document chunks
- `evaluation.py` – Evaluates chatbot responses
- `analyze_results.py` – Analyzes evaluation results
- `dataset/` – Stores PDF documents and CSV files
- `chroma_db/` – Stores vector database data

## Installation and Setup
1. Install Python 3.12.
2. Install Ollama.
3. Download the required models:
   - `ollama pull llama3.2`
   - `ollama pull nomic-embed-text`
4. Install project dependencies using `requirements.txt`.
5. Run the preprocessing and embedding scripts.
6. Start the Streamlit application.

## Run the Application
Run `streamlit run app.py` in the project terminal.

Open the local URL displayed in the terminal.

## Evaluation
The chatbot was tested using 20 questions. The evaluation results are stored in:
- `dataset/answers.csv`
- `dataset/evaluation_summary.csv`
- `dataset/evaluation_report.txt`

## Project Goal
To build a simple document-based AI study assistant that helps students understand Machine Learning concepts through interactive question answering.