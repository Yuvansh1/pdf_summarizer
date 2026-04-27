# PDF RAG Assistant

A local PDF question-answering app powered by **Ollama (llama3.2)** and **FastAPI**. Upload any PDF and chat with it — the entire document is indexed, not just the first few pages.

```
pdf_summarizer/
├── main.py            <- FastAPI backend
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── public/
    └── index.html     <- Frontend UI
```

## How it works

1. You upload a PDF. The backend extracts all text and splits it into overlapping chunks.
2. When you ask a question, the app finds the most relevant chunks using keyword matching.
3. Those chunks are sent to **llama3.2** via Ollama as context, and the answer is streamed back.

This means the whole document is searchable, not just the first 15,000 characters.

## Prerequisites

- [Python 3.9+](https://www.python.org/downloads/)
- [Ollama](https://ollama.com/download) installed and running
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed (for image-based PDFs)
- [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) with `bin` folder added to PATH
- llama3.2:1b model pulled:

```bash
ollama pull llama3.2:1b
```

## Setup

### 1. Install dependencies

```bash
pip install fastapi uvicorn python-multipart pypdf requests pytesseract pdf2image pillow
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 2. Make sure Ollama is running

```bash
ollama serve
```

Verify it works:

```bash
curl http://localhost:11434/api/tags
```

### 3. Start the backend

```bash
uvicorn main:app --reload --port 8080
```

### 4. Open the app

Open your browser and go to `http://localhost:8080`.

That's it. No separate frontend server needed — FastAPI serves the UI directly.

## Docker (optional)

```bash
docker compose up --build
```

The app will be available at `http://localhost:8080`.

Note: when running in Docker, Ollama must be reachable from the container. Update `OLLAMA_URL` in `main.py` to point to your host machine (e.g. `http://host.docker.internal:11434` on Mac/Windows).

## Notes

- Sessions are stored in memory and reset when the server restarts.
- Image-based and scanned PDFs are supported via OCR (requires Tesseract + Poppler).
- The model name is set via the `MODEL` variable at the top of `main.py`. Change it to any model you have pulled in Ollama (e.g. `mistral`, `phi3`, `gemma2`).
- OCR pages may take a few seconds each to process on first upload.
