# AI Knowledge Base with Gemini and HuggingFace Embeddings

This project allows you to build a local knowledge base from your documents (PDF, DOCX, TXT, CSV, JSON) and interact with it through a command-line interface. It uses HuggingFace embeddings for semantic search and Google's Gemini model for reasoning.

---

## Features

- Builds embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Supports PDF, DOCX, TXT, CSV, and JSON files
- Uses FAISS for efficient document similarity search
- Gemini reasoning via LangChain
- Interactive CLI chat that continues until the user exits
- Automatically generates a vectorstore (no manual setup required)

---

## Project Structure

main_directory/
│
├── build_knowledgebase.py # Builds and saves the FAISS vectorstore
├── main.py # Runs the interactive chat interface
├── requirements.txt # Project dependencies
├── .env # Environment variables (user-created)
└── vectorstore/ # Auto-generated FAISS index (ignored in Git)

## Setup

### 1. Clone the repository

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
### 2. Create a virtual environment
#### windows
`.venv\Scripts\activate`

#### linux/mac
source .venv/bin/activate

### 3. install dependendencies
`pip install -r requirements.txt`

### 4. Environment Setup

Create a .env file in your project’s root directory and add:
`DOCS_PATH=path/to/your/documents`

### 5. Building the Knowledge Base

Run the script below to process your documents and generate embeddings:
`python build_knowledgebase.py`
It will:
Load all supported documents from the folder you specified in .env
Split them into text chunks
Generate embeddings
Save a local FAISS vectorstore (in a folder called vectorstore/)
Do not copy the vectorstore folder to GitHub — it’s automatically generated when you build your knowledge base.

### 6. Running the main script
Once your knowledge base is built run:
```python main.py```
#### inspired by @chachacollins GATEKEEPER







