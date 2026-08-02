# Stage 6: LangChain hybrid-RAG agent

This CLI program uses a LangGraph ReAct agent, OpenRouter's
`nvidia/nemotron-3-nano-30b-a3b:free`, Hugging Face
`sentence-transformers/all-MiniLM-L6-v2` embeddings, and ChromaDB.

It exposes four LangChain tools: flight scheduling, hotel scheduling,
currency conversion, and hybrid retrieval. Retrieval uses reciprocal-rank
fusion to combine Chroma cosine similarity with in-process BM25 lexical
ranking. On every run, all readable files below `data/` are recursively
loaded, split, and indexed. PDF and DOCX files are supported explicitly; all
other non-binary files are read as UTF-8 text.

Only the user and final assistant messages are persisted in the conversation
memory collection. Intermediate agent/tool messages are never stored. The
LangGraph checkpointer manages the active graph conversation, while Chroma
retains conversation memory across separate CLI runs.

## Setup and use

1. Install Node.js 20 or newer and install Chroma locally with
   `pip install chromadb`. The script automatically starts a local persistent
   Chroma server when one is not already listening on `localhost:8000`.
2. Copy `.env-example` to `.env` and fill in both API keys.
3. Install and run:

   ```sh
   npm install
   node main.js "What internal information do we have about the conference?"
   ```

The complete persisted conversation and the final answer are printed to
standard output. Put files to retrieve in `data/`; nested folders are allowed.

Only `OPENROUTER_API_KEY` and `HF_API_KEY` are required. Optional settings all
have defaults: `LLM_MODEL_NAME`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_SSL`,
`THREAD_ID`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_COUNT`,
`AUTO_START_CHROMA`, and `CHROMA_COMMAND`. Set `AUTO_START_CHROMA=false` when
using a separately managed Chroma server.
