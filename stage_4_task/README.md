# Stage 4: JavaScript RAG API

A Retrieval-Augmented Generation (RAG) API built with JavaScript and Express, without LangChain.

The application:

- uploads and extracts text from context documents;
- creates sentence/paragraph-aware chunks using `CHUNK_LENGTH` as the maximum size;
- generates embeddings with Hugging Face model `sentence-transformers/all-MiniLM-L6-v2`;
- stores and searches embeddings in ChromaDB using cosine similarity;
- sends the retrieved context and question to Gemini 2.5 Flash;
- returns a grounded answer and the source filenames.

## Requirements

- Node.js 20 or newer
- npm
- valid Hugging Face and Gemini API keys
- a running ChromaDB server reachable through `CHROMA_DB_HOST` and `CHROMA_DB_PORT`

ChromaDB is configured as an external service because the task provides its host and port through environment variables. The application itself starts with only `npm install` and `node main.js` once that configured service is available.

## Installation

Open a terminal in this directory:

```powershell
cd "stage_4_task"
npm install
```

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Do not commit `.env`; it contains secret API keys and is excluded by `.gitignore`.

## Environment configuration

The implementation uses exactly the environment variables specified in the task:

```dotenv
HF_API_KEY=hf_your_real_token
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
GEMINI_API_KEY=your_real_gemini_api_key
LLM_MODEL_NAME=gemini-2.5-flash
CHROMA_DB_HOST=localhost
CHROMA_DB_PORT=8000
RAG_DATA_DIR=./data
CHUNK_LENGTH=1000
SERVER_PORT=3000
```

| Variable | Purpose |
| --- | --- |
| `HF_API_KEY` | Authenticates requests to the Hugging Face inference API. |
| `EMBED_MODEL_NAME` | Hugging Face sentence-transformer embedding model. |
| `GEMINI_API_KEY` | Authenticates requests to the Gemini API. |
| `LLM_MODEL_NAME` | Gemini model used to generate answers. |
| `CHROMA_DB_HOST` | Hostname of the configured ChromaDB server. |
| `CHROMA_DB_PORT` | Port exposed by the configured ChromaDB server. |
| `RAG_DATA_DIR` | Local directory where uploaded source files are saved. It is created automatically. |
| `CHUNK_LENGTH` | Maximum number of characters in each semantic chunk. |
| `SERVER_PORT` | Port used by the Express API. |

Restart `node main.js` whenever `.env` is changed.

## Starting ChromaDB for local testing

Skip this section when a working ChromaDB host and port have already been provided by the evaluator.

### Option A: Docker

```powershell
docker run --rm --name stage4-chromadb -p 8000:8000 chromadb/chroma
```

### Option B: Python

If Docker is unavailable but Python is installed:

```powershell
py -m pip install chromadb
chroma run --host localhost --port 8000 --path ".\chroma-data"
```

Keep the ChromaDB terminal running. Verify the server in another terminal:

```powershell
curl.exe http://localhost:8000/api/v2/heartbeat
```

For either local option, use:

```dotenv
CHROMA_DB_HOST=localhost
CHROMA_DB_PORT=8000
```

## Starting the API

From the `stage_4_task` directory, run:

```powershell
node main.js
```

Expected output:

```text
RAG API listening on port 3000
```

Leave this terminal running while testing the endpoints.

## Testing the endpoints

The recommended order is:

1. check `/health`;
2. upload at least one document through `/upload`;
3. ask a document-related question through `/prompt`.

The examples below use Windows PowerShell.

### 1. Health check

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:3000/health"
```

Expected HTTP status: `200`.

Expected body:

```json
{
  "status": "ok"
}
```

### 2. Upload documents

`POST /upload` accepts `multipart/form-data`. The file field must be named `files`; there are no other payload parameters.

Supported formats: PDF, DOCX, TXT, Markdown, CSV, JSON, HTML, and XML. A request can contain up to 20 files, with a maximum size of 25 MB per file.

Use the full path to an existing file:

```powershell
curl.exe -X POST http://localhost:3000/upload `
  -F "files=@C:\Users\NEW USER\Downloads\document.pdf"
```

Upload multiple documents by repeating the `files` field:

```powershell
curl.exe -X POST http://localhost:3000/upload `
  -F "files=@C:\Users\NEW USER\Downloads\document.pdf" `
  -F "files=@C:\Users\NEW USER\Downloads\notes.txt"
```

Example successful response:

```json
{
  "message": "Files uploaded and indexed successfully",
  "files": [
    {
      "filename": "document.pdf",
      "chunks": 8
    }
  ],
  "chunksIndexed": 8
}
```

Uploading performs document extraction, semantic chunking, Hugging Face embedding generation, and ChromaDB indexing. Therefore, ChromaDB and the Hugging Face API key must both be working.

### 3. Prompt the RAG system

Upload a document first. Then send JSON containing the required `query` field to `POST /prompt`.

PowerShell's native request command avoids JSON quoting problems:

```powershell
$body = @{
    query = "What is the main purpose of this project?"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:3000/prompt" `
    -ContentType "application/json" `
    -Body $body
```

Example successful response:

```json
{
  "answer": "The main purpose of the project is...",
  "sources": [
    "document.pdf"
  ]
}
```

The endpoint embeds the question, retrieves the five most similar chunks from ChromaDB, and asks Gemini to answer using only that retrieved context.

## Troubleshooting

### `Failed to connect to chromadb`

- Confirm that the ChromaDB server is running.
- Confirm that `CHROMA_DB_HOST` and `CHROMA_DB_PORT` match the server.
- For local testing, check `http://localhost:8000/api/v2/heartbeat`.
- Restart the Node application after changing `.env`.

### `Unauthorized`

The configured service requires authentication or an API key is invalid. The task only supplies ChromaDB host and port, so use a compatible ChromaDB server that accepts that connection. Chroma Cloud requires additional credentials that are not part of the task's required environment variables.

### `Expected property name or '}' in JSON`

PowerShell passed malformed JSON to the API. Use the `Invoke-RestMethod` example above instead of manually quoted JSON.

### `No indexed context is available`

Upload at least one readable document successfully before calling `/prompt`.

### Hugging Face or Gemini authorization errors

Confirm that `HF_API_KEY` and `GEMINI_API_KEY` contain real, active keys and restart the application.

## Project structure

```text
stage_4_task/
|-- main.js
|-- package.json
|-- package-lock.json
|-- .env.example
|-- .gitignore
|-- README.md
`-- data/                 # Created automatically; ignored by Git
```
