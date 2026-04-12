"""
app.py — Flask web server with RAG pipeline.

Serves the chat UI and handles user queries by:
1. Embedding the query
2. Retrieving top-k relevant chunks from Pinecone
3. Generating an answer with HuggingFace LLM + source citations

Usage:
    python app.py
"""
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os


app = Flask(__name__)


# ── Load environment variables ──────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["HF_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


# ── Initialize embeddings & vector store ────────────────────────────
embeddings = download_hugging_face_embeddings()

index_name = "impexp-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)


# ── Build RAG chain ────────────────────────────────────────────────
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    """Handle user messages — run RAG pipeline and return answer + sources."""
    msg = request.form["msg"]
    print(f"🔍 User: {msg}")

    response = rag_chain.invoke({"input": msg})

    answer = response["answer"]
    print(f"💬 Bot: {answer}")

    # Extract source info from retrieved documents
    sources = []
    if "context" in response:
        for doc in response["context"]:
            source_info = doc.metadata.get("source", "unknown")
            if source_info == "export_data":
                country = doc.metadata.get("country_to", "")
                commodity = doc.metadata.get("commodity", "")
                sources.append(f"Export Data: India → {country} ({commodity})")
            elif source_info == "book" or source_info != "export_data":
                page = doc.metadata.get("page", "")
                if page:
                    sources.append(f"Book — Page {int(page) + 1}")
                else:
                    sources.append("Book")

    # Deduplicate sources
    unique_sources = list(dict.fromkeys(sources))

    return jsonify({
        "answer": answer,
        "sources": unique_sources,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
