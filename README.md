# OptiRAG-Document-Intelligence-Semantic-Caching-Engine
# OptiRAG: Layout-Aware Document Intelligence & Semantic Caching Engine

An advanced Retrieval-Augmented Generation (RAG) pipeline designed to process highly technical engineering documents (e.g., aerospace textbooks, aerodynamic telemetry). This system utilizes layout-aware computer vision to preserve complex mathematical equations and tabular data, while implementing a semantic vector cache to reduce API latency by up to 90%.

**Author:** Chitharth

## System Architecture Highlights

* **Layout-Aware Chunking:** Bypasses naive character-limit chunking. Uses the `unstructured` library to run object-detection models over PDFs, identifying and preserving contiguous mathematical structures (like turbomachinery velocity triangles) and data tables before ingestion.
* **Semantic Caching Layer:** Intercepts user queries and converts them to mathematical vectors using `SentenceTransformers`. If a mathematically similar query was asked previously (cosine similarity > 85%), the system returns the cached answer instantly, bypassing the LLM.
* **FastAPI Integration:** Wraps the entire backend in a high-speed asynchronous REST API, ready for deployment and frontend integration.

## Core Tech Stack
* **LLM Engine:** Google Gemini API (`gemini-3.5-flash`)
* **Vector Database:** ChromaDB (Local persistent storage)
* **Embeddings:** HuggingFace `SentenceTransformers` (`all-MiniLM-L6-v2`)
* **Document Parsing:** Unstructured.io (Vision/Layout-aware chunking)
* **Web Framework:** FastAPI & Uvicorn

## Installation & Setup

1. **Clone and create a virtual environment:**
   ```bash
   git clone [https://github.com/yourusername/optirag-engine.git](https://github.com/yourusername/optirag-engine.git)
   cd optirag-engine
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
