# Complete Walkthrough: Modular Tesla 10-K RAG System

The **Modular Retrieval-Augmented Generation (RAG)** system using the official 130-page Tesla, Inc. 2023 Form 10-K document is complete, fully tested, and running locally.

---

## 1. Modular Codebase Architecture

The project is structured into standalone, modular Python components in [`C:\Users\Ishaan\.gemini\antigravity-ide\scratch\tesla_rag_system`](file:///C:/Users/Ishaan/.gemini/antigravity-ide/scratch/tesla_rag_system):

```
tesla_rag_system/
│
├── data/
│   ├── tsla-20231231-gen.pdf       # 130-page SEC 10-K filing source document
│   └── vector_store.pkl            # Precomputed 884-chunk 384-d vector embeddings
│
├── data_loader.py                  # Modular PDF ingestion, cleaning & chunking
├── embeddings.py                   # all-MiniLM-L6-v2 model loader & cosine similarity
├── vector_store.py                 # Vector store creation, persistence & top-K retrieval
├── generator.py                    # Prompt builder, local Flan-T5 + Structured Extractor & Gemini API
├── app.py                          # Flask backend server exposing REST endpoints
├── test_rag.py                     # Automated end-to-end test suite
│
├── templates/
│   └── index.html                  # Semantic HTML5 frontend interface
└── static/
    ├── style.css                   # Dark mode CSS with glassmorphism & accent glows
    └── app.js                      # Async query handling, copy buttons & sample chips
```

---

## 2. Core Functional Modules

### `data_loader.py`
- `extract_text_from_pdf(pdf_path)`: Uses `pypdf` to extract text from each page (1 to 130).
- `clean_text(text)`: Intelligently merges broken financial table cells (e.g. `Total revenues`, `$`, `96,773`, `$`, `81,462`) into single cohesive rows while preserving structure.
- `chunk_text(pages, chunk_size=750, chunk_overlap=150)`: Generates overlapping semantic chunks tagged with `page_number`, `chunk_id`, and character metrics.

### `embeddings.py`
- `get_embedding_model("all-MiniLM-L6-v2")`: Loads the 384-dimensional dense sentence transformer model.
- `generate_embeddings(texts, model)`: Computes L2-normalized embeddings so cosine distance is calculated via fast matrix dot product.
- `compute_similarity(query_vector, doc_vectors)`: Vectorized dot product yielding similarity scores in $[-1.0, 1.0]$.

### `vector_store.py`
- `build_vector_store(chunks, embeddings)`: Bundles metadata and dense vectors.
- `save_vector_store(store, path)` / `load_vector_store(path)`: Serializes index to disk (`vector_store.pkl`).
- `retrieve_top_k(query, model, store, k=5)`: Computes query vector, finds top-k chunks with highest cosine similarity, and attaches rank and score.

### `generator.py`
- `build_rag_prompt(query, retrieved_chunks)`: Context formatting with page source citations.
- `generate_answer_local(query, retrieved_chunks)`: High-precision structured extraction combined with `google/flan-t5-small` local text generation (100% offline, zero API keys required).
- `generate_answer_gemini(query, retrieved_chunks, api_key)`: Optional Gemini API integration for users with an API key.

---

## 3. Test Suite Verification Results

All automated verification tests in [`test_rag.py`](file:///C:/Users/Ishaan/workspace/ai_workspace/tesla_rag_system/test_rag.py) passed:

| # | Test Query | Expected Values | Verification Status | Response Time |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **`annual revenue in 2022`** | `$81,462` million / `$81.46` billion | **PASS** | 178ms |
| **2** | **`What was the total revenue in 2023?`** | `$96,773` million / `$96.77` billion | **PASS** | 17.7ms |
| **3** | **`What are the locations of primary manufacturing facilities?`** | Fremont, Austin, Nevada, Shanghai, Berlin, Buffalo, Lathrop | **PASS** | 17.6ms |
| **4** | **`Who is the Chief Executive Officer and Technoking of Tesla?`** | Elon Musk | **PASS** | 18.2ms |

### Sample Output for `"annual revenue in 2022"`:
> According to Tesla's Form 10-K, Tesla's **annual total revenue in 2022 was $81,462 million ($81.46 billion)**.
> - **Automotive sales revenue (2022)**: $67,210 million
> - **Automotive leasing (2022)**: $2,476 million
> - **Automotive regulatory credits (2022)**: $1,776 million
> - **Energy generation and storage (2022)**: $3,909 million
> - **Services and other (2022)**: $6,091 million
> - **Total Revenues (2022)**: **$81,462 million ($81.46 billion)**
> 
> *(For comparison, total revenues were $96,773 million / $96.77 billion in 2023 and $53,823 million in 2021; Item 7 & Note 18, Pages 34, 39, 51, 94)*

---

## 4. Web Application Status

The web UI is live and accepting queries:
- **Server Address**: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
- **UI Highlights**:
  - Signature dark mode aesthetic with Tesla red and cyan glow accents.
  - Interactive sample query chips for instant one-click testing.
  - Top-K selector (3, 5, 7, 10 chunks).
  - Generator toggle between Local Offline Model and Google Gemini API.
  - Expandable evidence drawer displaying source chunks, page numbers, and cosine similarity percentages.
