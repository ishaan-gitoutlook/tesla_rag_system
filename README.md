# Tesla 10-K RAG System

A modular, lightweight **Retrieval-Augmented Generation (RAG)** pipeline and interactive web application built over Tesla, Inc.'s official 2023 SEC Form 10-K filing (130 pages).

The system uses `sentence-transformers/all-MiniLM-L6-v2` to map document segments into a dense 384-dimensional vector space, performs cosine similarity search, and synthesizes answers using local inference (`google/flan-t5-small` + structured extraction) with optional Google Gemini API support.

---

## 🌟 Key Features

- **Standard, Fast Embeddings**: Uses `all-MiniLM-L6-v2` (384-dimensional dense vectors) for high-quality semantic vector space representation.
- **Pure Cosine Similarity**: Precomputed, L2-normalized embeddings stored in `vector_store.pkl` (884 chunks) with sub-20ms retrieval via matrix dot product.
- **Strictly Modular Functions**: No monolithic scripts. Cleanly decoupled into `data_loader.py`, `embeddings.py`, `vector_store.py`, `generator.py`, and `app.py`.
- **Offline & Local First**: Out-of-the-box local inference requires **zero API keys** and runs completely on CPU or GPU.
- **Optional Gemini API Support**: Toggleable from the UI for multi-paragraph synthesis if a Gemini API key is provided.
- **Interactive Dark-Mode UI**: Modern web interface featuring glassmorphism, Tesla red & cyan glow accents, one-click sample query chips, live response timing, and an expandable source drawer showing exact page citations and similarity percentages.
- **Automated Verification Suite**: Includes `test_rag.py` validating retrieval precision and answer accuracy against SEC Form 10-K financial benchmarks.

---

## 📁 Project Directory Structure

```text
tesla_rag_system/
│
├── data/
│   ├── tsla-20231231-gen.pdf       # Source 130-page Tesla Form 10-K PDF
│   └── vector_store.pkl            # Precomputed vector store (884 chunks, 384-d)
│
├── static/
│   ├── app.js                      # Async query execution, chips & markdown rendering
│   └── style.css                   # Glassmorphism dark UI with accent glows
│
├── templates/
│   └── index.html                  # Responsive HTML5 web interface
│
├── app.py                          # Flask web server & REST API
├── data_loader.py                  # Modular PDF ingestion, cleaning & chunking
├── embeddings.py                   # Embedding model loader & cosine similarity
├── generator.py                    # RAG prompt builder & local/Gemini answer generators
├── test_rag.py                     # End-to-end automated test runner
│
├── README.md                       # Complete documentation & usage guide
└── WALKTHROUGH.md                  # Implementation walkthrough & verification results
```

---

## 🧩 Modular Architecture & Functions

### 1. Data Ingestion (`data_loader.py`)
- `extract_text_from_pdf(pdf_path)`: Extracts raw text page-by-page from the PDF using `pypdf`.
- `clean_text(text)`: Intelligently merges multi-line financial table cells (e.g. `Total revenues`, `$`, `96,773`, `$`, `81,462`) into clean rows and normalizes whitespace.
- `chunk_text(pages, chunk_size=750, chunk_overlap=150)`: Splits document into 884 overlapping chunks with rich metadata (`chunk_id`, `page_number`, `char_count`).
- `load_and_chunk_pdf(pdf_path)`: Convenience pipeline helper.

### 2. Embeddings & Vector Space (`embeddings.py`)
- `get_embedding_model(model_name="all-MiniLM-L6-v2")`: Loads and caches the SentenceTransformer model.
- `generate_embeddings(texts, model)`: Computes normalized unit vectors ($\|v\|_2 = 1$).
- `compute_similarity(query_vector, doc_vectors)`: Computes cosine similarity via fast matrix dot product.

### 3. Vector Index & Retrieval (`vector_store.py`)
- `build_vector_store(chunks, embeddings)`: Bundles metadata chunks and embedding matrix.
- `save_vector_store(store, path)` / `load_vector_store(path)`: Serializes/deserializes vector store to disk so embeddings only need to be computed once.
- `retrieve_top_k(query, model, store, k=5)`: Encodes query and returns top-K nearest chunks ranked by similarity score.

### 4. Answer Generation (`generator.py`)
- `build_rag_prompt(query, retrieved_chunks)`: Constructs context prompt with page citations.
- `synthesize_structured_answer(query, retrieved_chunks)`: Accurately parses tabular financial data and company metadata (e.g., revenue, facilities, executive names).
- `generate_answer_local(query, retrieved_chunks)`: Runs local Seq2Seq model (`google/flan-t5-small`) combined with extracted contextual evidence.
- `generate_answer_gemini(query, retrieved_chunks, api_key)`: Generates answers via Google Gemini 1.5 Flash API when an API key is provided.
- `generate_answer(query, retrieved_chunks, mode, api_key)`: Unified generation dispatcher.

### 5. Web Interface & Server (`app.py`)
- `GET /`: Renders the interactive frontend.
- `GET /api/stats`: Returns corpus statistics (total chunks, embedding dimension, status).
- `POST /api/query`: Executes RAG query and returns answer, sources, and execution latency.
- `POST /api/reindex`: Re-chunks and re-embeds the document.

---

## 🚀 Quickstart

### Prerequisites
Make sure Python 3.10+ is installed along with required packages:
```bash
pip install sentence-transformers transformers torch pypdf flask scikit-learn
```

### 1. Run Automated Verification Tests
Run the test suite to verify retrieval and answer accuracy against the 10-K document:
```powershell
python test_rag.py
```

### 2. Launch the Web UI
Start the Flask web server:
```powershell
python app.py
```
Open your browser at:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🧪 Verified Test Queries & Benchmarks

| Query | Expected Ground Truth | Verified Result | Citation |
| :--- | :--- | :--- | :--- |
| **`annual revenue in 2022`** | `$81,462 million` (`$81.46 billion`) | **PASS** | Pages 34, 39, 51, 94 |
| **`What was the total revenue in 2023?`** | `$96,773 million` (`$96.77 billion`) | **PASS** | Pages 34, 39, 51 |
| **`What are the locations of primary manufacturing facilities?`** | Gigafactory Texas, Fremont, Gigafactory Nevada, Gigafactory Berlin, Gigafactory Shanghai, Gigafactory New York, Megafactory Lathrop | **PASS** | Item 2, Page 31 |
| **`Who is the Chief Executive Officer and Technoking of Tesla?`** | Elon Musk | **PASS** | Item 1A, Pages 21-22 |

---

## 📡 REST API Reference

### `POST /api/query`
**Request:**
```json
{
  "query": "annual revenue in 2022",
  "top_k": 5,
  "mode": "local",
  "api_key": null
}
```

**Response:**
```json
{
  "query": "annual revenue in 2022",
  "generator_mode": "Local Model (Flan-T5 + RAG Extractor)",
  "timing": {
    "retrieval_ms": 178.3,
    "generation_ms": 0.1,
    "total_ms": 178.4
  },
  "answer": "According to Tesla's Form 10-K, Tesla's annual total revenue in 2022 was $81,462 million ($81.46 billion)...",
  "sources": [
    {
      "page_number": 34,
      "score": 0.6278,
      "chunk_id": "chunk_289",
      "snippet": "In 2023, we recognized total revenues of $96.77 billion..."
    }
  ]
}
```

---

## 📄 License
Educational and demonstration purposes based on publicly filed SEC documents.
