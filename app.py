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
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
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
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


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
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3,
    huggingfacehub_api_token=HF_TOKEN,
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
    try:
        msg = request.form.get("msg", "")
        if not msg:
            return jsonify({"answer": "Please enter a message.", "sources": []})

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
                elif source_info == "laws_kb":
                    sources.append("Trade Laws & Regulations KB")
                elif source_info == "book":
                    page = doc.metadata.get("page", "")
                    if page:
                        sources.append(f"Book — Page {int(page) + 1}")
                    else:
                        sources.append("Book")
                else:
                    sources.append("Knowledge Base")

        # Deduplicate sources
        unique_sources = list(dict.fromkeys(sources))

        return jsonify({
            "answer": answer,
            "sources": unique_sources,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error in /get route: {e}")
        return jsonify({
            "answer": f"Sorry, an error occurred: {str(e)}",
            "sources": [],
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
