from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List
import pandas as pd
import os


# ---------- Book (PDF) Loading ----------

def load_pdf_file(data_dir: str) -> List[Document]:
    """
    Load all PDF files from a directory using LangChain's DirectoryLoader.
    Returns a list of Document objects with page_content and metadata.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# ---------- Export CSV Loading ----------

def load_csv_as_documents(csv_path: str) -> List[Document]:
    """
    Read the merged_country_wise.csv and convert each row into a
    LangChain Document for embedding and retrieval.

    CSV columns: country_from, country_to, S.No., HSCode, Commodity,
                 2023-2024, 2024-2025, %Growth
    """
    df = pd.read_csv(csv_path)

    documents: List[Document] = []
    for _, row in df.iterrows():
        country_from = str(row.get("country_from", "India")).strip()
        country_to = str(row.get("country_to", "")).strip()
        hs_code = str(row.get("HSCode", "")).strip()
        commodity = str(row.get("Commodity", "")).strip()
        value_2023_24 = str(row.get("2023-2024", "n/a")).strip()
        value_2024_25 = str(row.get("2024-2025", "n/a")).strip()
        pct_growth = str(row.get("%Growth", "n/a")).strip()

        # Build human-readable text for embedding
        text = (
            f"Indian Export Data: {country_from} exports {commodity} "
            f"(HS Code: {hs_code}) to {country_to}. "
            f"Export value in 2023-2024: USD {value_2023_24} million. "
            f"Export value in 2024-2025: USD {value_2024_25} million. "
            f"Growth: {pct_growth}%."
        )

        metadata = {
            "source": "export_data",
            "country_from": country_from,
            "country_to": country_to,
            "hs_code": hs_code,
            "commodity": commodity,
        }

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


# ---------- Metadata Filtering ----------

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Strip heavy metadata from documents, keeping only essential fields:
    source, chapter (for book), country_to (for export data).
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        meta = {}
        # Preserve key metadata fields
        for key in ["source", "country_from", "country_to", "hs_code", "commodity", "page"]:
            if key in doc.metadata:
                meta[key] = doc.metadata[key]
        # Tag book documents
        if "source" not in meta:
            meta["source"] = "book"

        minimal_docs.append(
            Document(page_content=doc.page_content, metadata=meta)
        )
    return minimal_docs


# ---------- Text File Loading ----------

def load_text_file(file_path: str) -> List[Document]:
    """
    Load a plain text file and return it as a list of LangChain Documents.
    Each document is tagged with source='laws_kb' in metadata.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    # Tag each document with a 'laws_kb' source
    for doc in documents:
        doc.metadata["source"] = "laws_kb"
    return documents


# ---------- Text Chunking ----------

def text_split(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Book documents get chunked (500 chars, 50 overlap).
    Laws/KB documents get chunked (800 chars, 100 overlap) to preserve section context.
    Export data documents are already compact per-row, so they pass through
    with a larger chunk size to avoid splitting single entries.
    """
    book_docs = [d for d in documents
                 if d.metadata.get("source") not in ("export_data", "laws_kb")]
    laws_docs = [d for d in documents if d.metadata.get("source") == "laws_kb"]
    export_docs = [d for d in documents if d.metadata.get("source") == "export_data"]

    # Chunk book documents
    book_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    book_chunks = book_splitter.split_documents(book_docs)

    # Chunk laws/KB documents — larger chunks to preserve structured sections
    laws_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    laws_chunks = laws_splitter.split_documents(laws_docs)

    # Export rows are typically short, split only if very long
    export_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    export_chunks = export_splitter.split_documents(export_docs)

    return book_chunks + laws_chunks + export_chunks


# ---------- Embeddings ----------

def download_hugging_face_embeddings():
    """
    Download and return HuggingFace sentence-transformer embeddings.
    Model: all-MiniLM-L6-v2 (384 dimensions).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
