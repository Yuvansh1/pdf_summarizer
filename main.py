import os
import requests
import uuid
import io
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "llama3.2:1b"  # much smaller, faster version

# Chunk size in characters. 6000 chars per chunk keeps prompts manageable.
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 200  # overlap so context is not lost at boundaries

# In-memory session storage: session_id -> list of text chunks
sessions = {}


def chunk_text(text: str) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def find_relevant_chunks(chunks: list, query: str, top_n: int = 4) -> str:
    """Return top_n most relevant chunks using simple keyword matching."""
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [chunk for _, chunk in scored[:top_n]]
    return "\n---\n".join(selected)


def ask_llm(prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        print(f"Ollama Error: {e}")
        return f"AI Error: Make sure Ollama is running with '{MODEL}'. ({str(e)})"


# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "public", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return "<h1>404: index.html not found</h1>"


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs are supported")

    try:
        session_id = str(uuid.uuid4())
        content = await file.read()
        pdf_reader = PdfReader(io.BytesIO(content))

        # Try normal text extraction first
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # If little/no text found, fall back to OCR
        if len(text.strip()) < 100:
            print("Low text detected, falling back to OCR...")
            images = convert_from_bytes(content)
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"

        if not text.strip():
            raise HTTPException(status_code=422, detail="Could not extract any text.")

        chunks = chunk_text(text)
        sessions[session_id] = chunks

        return {
            "session_id": session_id,
            "status": "ready",
            "pages": len(pdf_reader.pages),
            "chunks": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query")
    session_id = data.get("session_id")

    if not query or not session_id:
        raise HTTPException(status_code=400, detail="Missing query or session_id")

    chunks = sessions.get(session_id)
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please re-upload your PDF."
        )

    context = find_relevant_chunks(chunks, query, top_n=2)

    prompt = f"""You are a helpful assistant. Use ONLY the document excerpts below to answer the user's question. \
If the answer is not in the excerpts, say so clearly.

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{query}

ANSWER:"""

    answer = ask_llm(prompt)
    return {
        "answer": answer,
        "sources": [{"id": 1, "preview": "Extracted from uploaded PDF"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
