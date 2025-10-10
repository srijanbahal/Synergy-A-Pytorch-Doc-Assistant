# Synergistic PyTorch RAG Assistant (SPRA)

**SPRA is a context-aware AI developer tool, delivered as a Chrome Extension, designed to provide expert-level assistance for the PyTorch deep learning framework. It uses a novel hybrid RAG architecture to deliver accurate, conceptually rich, and citation-backed answers.**

This project moves beyond traditional RAG by fusing multiple advanced techniques to create a more robust and intelligent system that can act as a learning tool, a debugging assistant, and a library navigator.

---

## Architecture: Synergistic Hybrid RAG

The core of SPRA is a unique pipeline that combines the strengths of several state-of-the-art RAG architectures.

![Architecture Diagram](https://i.imgur.com/your-diagram-url.png) 1.  **Adaptive Routing:** A query classifier first determines if a question is simple (for a fast vector search) or complex (requiring the full pipeline).
2.  **Hybrid Retrieval:** The system performs two types of searches in parallel:
    * **Graph-Based Search (Neo4j):** Queries a Knowledge Graph of PyTorch to understand the conceptual relationships between modules, functions, and classes.
    * **Vector Search (ChromaDB):** Performs semantic search over the documentation text to find specific code examples and detailed explanations.
3.  **Corrective Evaluation (CRAG):** An evaluation step checks the relevance of the retrieved context. If quality is low, it can trigger a fallback search.
4.  **Reflective Generation (Self-RAG):** The final context is passed to an LLM with instructions to generate an answer based *only* on the provided information and to cite its sources, drastically reducing hallucination.

---

## Features

This project implements a wide range of advanced RAG techniques:

* **Query Transformation:** Utilizes **Multi-Query Generation** and **RAG-Fusion** to broaden the search scope and re-rank results for higher relevance.
* **Intelligent Routing:** Employs **Logical and Semantic Routing** to direct queries to the appropriate internal logic.
* **Advanced Indexing:** Builds a **Knowledge Graph** to capture conceptual relationships, going beyond simple text-chunk embeddings.
* **Self-Correction & Citation:** Implements principles from **CRAG** and **Self-RAG** to ensure answers are accurate and directly tied to the source documentation.
* **End-to-End Evaluation:** Uses the **RAGAS framework** to objectively measure performance across metrics like Faithfulness, Answer Correctness, and Context Recall.

---

## Tech Stack

* **Backend:** Python, FastAPI, Docker
* **AI/RAG:** LangChain, OpenAI, Sentence-Transformers
* **Data Stores:** ChromaDB (Vector Store), Neo4j (Graph Database)
* **Frontend:** JavaScript, HTML, CSS (Chrome Extension)
* **Evaluation:** RAGAS, DeepEval

---

## Setup & Installation

**(You will fill this out as you build the project)**

1.  **Backend Setup**
    ```bash
    cd backend
    pip install -r requirements.txt
    # Add steps for setting up .env file, etc.
    uvicorn app.main:app --reload
    ```
2.  **Chrome Extension Setup**
    * Navigate to `chrome://extensions` in your browser.
    * Enable "Developer mode".
    * Click "Load unpacked" and select the `chrome_extension` directory.