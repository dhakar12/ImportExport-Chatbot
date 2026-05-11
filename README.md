# Import/Export Assistant

A project that builds an Import/Export Business chatbot using a Retrieval-Augmented Generation (RAG) pipeline. The system combines procedural guidance from a trade business knowledge base with Indian export data, and returns answers with source citations.

---

## 1. Introduction

### 1.1 Context and motivation

Entrepreneurs and businesses involved in import and export need both procedural guidance and reliable trade statistics. This project creates a chatbot that answers questions from a curated knowledge base: a standard import/export business reference and Indian export data from TRADESTAT-style sources. The system uses RAG so answers are grounded in sources and can be cited, rather than relying only on the language model’s training.

### 1.2 Importance of import–export trade

International trade connects countries, firms, and consumers. Imports bring goods and inputs that may be cheaper, higher quality, or unavailable domestically. Exports open new markets, earn foreign exchange, and support jobs and growth. For India, trade activity is central to economic development and global integration. For entrepreneurs, import–export offers the chance to scale beyond local markets but requires understanding regulations, logistics, finance, and demand.

### 1.3 Retrieval-Augmented Generation (RAG) — brief explanation

RAG combines search with language generation. The user question is used to retrieve relevant passages from a stored collection of documents or data, then those passages are passed to a language model as context. The model generates an answer grounded in the retrieved context, which reduces hallucinations and allows source citations.

### 1.4 How RAG can help entrepreneurs in import–export

RAG helps by:

- answering "how-to" questions from trusted guides and manuals,
- answering "what are the numbers?" questions from official trade statistics,
- showing where the answer came from,
- staying aligned with your selected knowledge base.

### 1.5 Objectives

- Provide a single conversational interface for procedural guidance and trade-data questions.
- Ground answers in designated documents and datasets.
- Use Indian export statistics so trade-data answers are authoritative.
- Design the pipeline to support future data updates and new source types.

### 1.6 System overview

The chatbot uses a knowledge base composed of:

- a trade business reference book,
- Indian export data in a merged CSV,
- trade law and regulation knowledge.

The system supports entrepreneur-style questions, trade-data queries, and general import/export questions.

### 1.7 Document structure

This README is organised into: introduction, system flow, data preparation, ingestion, query pipeline, web application, implementation status, advantages, applications, future scope, and sample conversations.

---

## 2. High-Level System Flow

The system is organised into four stages:

| Stage | Name | Mode | Description |
|:-----:|------|------|-------------|
| **0** | Data Preparation | Offline / Admin | Acquire and prepare source documents and export data. |
| **1** | Data & Ingestion | Offline / Admin | Parse, chunk, embed, and store content in a vector database. |
| **2** | Query → Answer | Online | Retrieve relevant chunks and generate answers with citations. |
| **3** | Web Application | Online | Chat UI and backend API that invoke the RAG pipeline. |

**End-to-end flow:**

```
User → Web UI → Backend API → RAG (Retrieve + LLM) → Answer + Citations → User
```

**Offline pipeline:**

```
Data Preparation → Parse → Chunk → Enrich → Embed → Store (Vector DB)
```

**Online pipeline:**

```
User Question → Embed Query → Retrieve Top-k → Build Context → LLM → Answer + Sources
```

---

## 3. Stage 0: Data Preparation

### 3.1 Purpose

Collect, structure, and organise source materials before ingestion. This stage ensures the book and export data are in a consistent form.

### 3.2 Process

| Step | Description |
|------|-------------|
| **Source acquisition** | Obtain the book text and Indian export data. |
| **Schema / structure** | Ensure export data has `country_from`, `country_to`, `HSCode`, `Commodity`, year value columns, and growth. |
| **Data cleaning** | Normalise text and validate numeric export values. |
| **Organisation** | Keep book text under `data/book/`, export CSV under `data/export/`, and law KB under `data/`. |

### 3.3 Outputs

- Book reference content ready for chunking.
- Merged Indian export CSV ready for ingestion.

### 3.4 Extensibility

New export data or book sections can be added and re-ingested without retraining the model.

---

## 4. Stage 1: Data & Ingestion

### 4.1 Purpose

Convert raw sources into searchable chunks and store them with embeddings in a vector database.

### 4.2 Process

| Step | Description |
|------|-------------|
| **Parse** | Extract text from book sources and load export CSV rows. |
| **Chunk** | Split book text into digestible passages; create data chunks for export rows or grouped country/commodity segments. |
| **Enrich** | Attach metadata for source type, chapter, country, HS code, and commodity. |
| **Embed** | Generate embeddings for each chunk using a shared model. |
| **Store** | Save chunks and embeddings in a vector database such as Pinecone. |

### 4.3 Outputs

- A vector database comprised of book chunks and export data chunks.
- Metadata to distinguish book content from export data.

### 4.4 Extensibility

Additional export years, new countries, or future dispute data can be added to the same pipeline.

---

## 5. Stage 2: Query → Answer

### 5.1 Purpose

For each user query, retrieve relevant context from the knowledge base and generate an answer with citations.

### 5.2 Process

| Step | Description |
|------|-------------|
| **Query embedding** | Convert the user question into an embedding. |
| **Retrieval** | Search the vector database for top-k relevant chunks. |
| **Context building** | Format retrieved chunks with source labels. |
| **Generation** | Send the user question plus retrieved context to the LLM. |
| **Response** | Return structured output with `answer` and `sources`. |

### 5.3 Outputs

- Answer text grounded in the knowledge base.
- Source citations for verification.

---

## 6. Stage 3: Web Application Flow

### 6.1 Purpose

Expose the RAG pipeline through a user-facing chat interface.

### 6.2 Process

| Step | Description |
|------|-------------|
| **User opens app** | The chat UI loads in the browser. |
| **User submits question** | The frontend sends the message to the backend. |
| **Backend processing** | The backend runs retrieval and LLM generation. |
| **Frontend display** | The answer and sources appear in the chat. |
| **Follow-up** | Multi-turn dialogue can be supported via chat history. |

### 6.3 Architecture

- Frontend: `templates/chat.html` and `static/style.css`
- Backend: `app.py` with RAG logic and API endpoint
- Retrieval engine: Pinecone vector store
- LLM: HuggingFace endpoint via `langchain-huggingface`

---

## 7. Summary

| Stage | Main output |
|:-----:|-------------|
| **0** | Prepared book and export data sources |
| **1** | Vector DB with embeddings |
| **2** | Retrieved answer with citations |
| **3** | User-facing chat app |

This system combines trusted documentation and official trade data into one conversational assistant.

---

## 8. Implementation Updates

### 8.1 Data sources in this repository

- **Book knowledge**: `data/book/` (book text or extracted content)
- **Indian export data**: `data/export/merged_country_wise.csv`
- **Trade laws KB**: `data/import_export_laws_knowledge_base.txt`

### 8.1.1 Public dataset on Kaggle

The merged Indian export dataset has also been published on Kaggle under a CC0 license. Anyone can download and reuse it from:

- https://www.kaggle.com/datasets/udaydhakad/export-data-of-india-2024-2025

This makes the dataset available publicly for research and reuse.

### 8.2 Current status

The repository includes the chat application and the data sources needed for the RAG pipeline. The app loads embeddings, connects to Pinecone, and serves the chat UI.

---

## 9. Result Update

### 9.1 Dataset summary

The export dataset is prepared as a merged CSV with columns such as:

- `country_from`
- `country_to`
- `S.No.`
- `HSCode`
- `Commodity`
- `2023-2024`
- `2024-2025`
- `%Growth`

### 9.2 Cleaning and preparation

The data is normalised for missing values, dropped summary rows, and standardised for ingestion.

### 9.3 Next steps

- finish ingestion of book and export data
- complete retrieval and LLM integration
- improve the web UI and deploy the app

---

## 10. Advantages

- Grounded answers from curated sources
- Source citations for transparency
- Official-style Indian export statistics
- One interface for both procedural guidance and trade data
- Easy to extend with new datasets

---

## 11. Applications

- Entrepreneur guidance on export/import business setup
- Trade market research with country and commodity figures
- Educational support for trade and customs topics
- Future extension to dispute guidance and case-based support

---

## 12. Workflow for Remaining Work

1. Parse and chunk book + export data.
2. Embed chunks and store in Pinecone.
3. Build the query pipeline with retrieval, prompt assembly, and LLM generation.
4. Finalise the chat UI and deploy.

---

## 13. Future Scope

- Add dispute data and case references
- Add multilingual support
- Add voice and chatbot analytics
- Integrate with real trade portals or customs APIs

---

## 14. Sample Conversation

**User:** How do I start an export business?

**Chatbot:**
1. Research product-market fit.
2. Register your business and obtain IEC.
3. Source quality suppliers and arrange logistics.
4. Comply with customs, certificates, and export regulations.
5. Find buyers and negotiate contracts.

*Source: trade business knowledge base and export guidance.*

**User:** What does India export to Australia?

**Chatbot:**
India exports commodities such as cereals and processed food products to Australia. The system can cite export data by HS code and value for FY 2023–2024 and 2024–2025.

*Source: Indian export data, `data/export/merged_country_wise.csv`.*

---

## Setup and Run

```bash
cd /Users/udaydhakar12/Documents/project_impexp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8080`.

## Environment variables

Create `.env` with:

```env
PINECONE_API_KEY=your_pinecone_api_key
HUGGINGFACEHUB_ACCESS_TOKEN=your_huggingface_api_token
```

## Screenshots

![Dashboard screenshot](docs/dashboard.png)

![Chat query screenshot](docs/scr1.png)

![Answer details screenshot](docs/scr2.png)

![Export data summary screenshot](docs/scr3.png)

![Conclusion and citations screenshot](docs/scr4.png)

---

## Important files

- `app.py` — Flask backend and RAG pipeline
- `src/prompt.py` — system prompt and response rules
- `src/helper.py` — embedding setup
- `templates/chat.html` — chat interface
- `static/style.css` — UI styling
- `data/export/merged_country_wise.csv` — export dataset
- `data/import_export_laws_knowledge_base.txt` — regulatory knowledge
- `docs/` — screenshot assets

## Notes

This README is inspired by the project report structure and shows how the system flow maps to the current repository. The screenshots are included to visualise the app experience and results.
