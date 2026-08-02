# Stage 6: LangChain hybrid-RAG agent

This project is a command-line LangChain agent built with JavaScript. It uses:

- LangGraph for the ReAct agent loop and active-conversation checkpointing;
- OpenRouter's `nvidia/nemotron-3-nano-30b-a3b:free` for reasoning;
- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` for embeddings;
- ChromaDB for document vectors and persistent conversation memory;
- hybrid retrieval combining Chroma cosine similarity with BM25 lexical search.

It does not expose an HTTP endpoint or use FastAPI or Express. The complete
conversation history for the active thread and the final answer are printed to
standard output.

## Agent tools

The agent can choose from four LangChain tools:

1. `get_flight_booking_schedule` returns an illustrative Lagos-to-Nairobi
   round-trip schedule and price.
2. `get_hotel_booking_schedule` returns an illustrative Nairobi hotel schedule
   and price.
3. `convert_currency` converts amounts using fixed illustrative rates.
4. `query_internal_information` searches documents in `data/` and relevant
   conversation memories using hybrid retrieval.

Travel schedules, prices, and exchange rates are demonstration data, not live
quotes.

## Requirements

- Node.js 20 or newer
- Python with ChromaDB installed:

  ```sh
  pip install chromadb
  ```

- An OpenRouter API key
- A Hugging Face API key with access to the inference API

The JavaScript `chromadb` dependency is a client. The Python installation
provides the local Chroma server and its `chroma` command.

## Setup

From the `stage_6_task` directory, install the JavaScript dependencies:

```sh
npm install
```

Copy `.env-example` to `.env`:

```sh
# macOS/Linux
cp .env-example .env

# Windows Command Prompt
copy .env-example .env
```

Set both required values in `.env`:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key_here
HF_API_KEY=your_huggingface_api_key_here
```

Do not commit the `.env` file. It is excluded by `.gitignore`.

## Adding internal documents

Place documents anywhere below `data/`. Nested directories are scanned
recursively. The knowledge collection is rebuilt on every invocation, so new,
edited, and removed files are reflected automatically.

Supported content:

- PDF files containing extractable text;
- DOCX files;
- UTF-8 text files such as TXT, Markdown, CSV, JSON, HTML, XML, and source code.

Image-only/scanned PDFs require OCR first. Images, audio, video, XLSX, legacy
DOC files, and other binary formats are not parsed. Hidden files and folders
whose names begin with `.` are ignored.

The included `data/.gitkeep` file only allows Git to retain the otherwise-empty
directory. It is ignored by the indexer and can remain after documents are
added.

If `data/` has no documents, the agent has no internal document knowledge and
will correctly report that nothing relevant was found. Information from the
previous stage is not imported automatically.

## Running the agent

Pass the prompt as a command-line argument:

```sh
node main.js "What internal information do we have about the global warming?"
```

For the illustrative travel tools:

```sh
node main.js "Calculate the round-trip flight time and three-night logistics cost from Lagos to Nairobi, then convert the total to NGN."
```

The script checks `localhost:8000` and automatically starts a persistent local
Chroma server when necessary. Chroma data is stored in `.chroma/`, which is
excluded from Git. To run Chroma yourself instead, use:

```sh
chroma run --path .chroma --host localhost --port 8000
```

Then set `AUTO_START_CHROMA=false` if desired.

## Conversation memory

Only user prompts and final assistant responses are saved in Chroma. Internal
agent reasoning, intermediate assistant tool requests, and tool results are not
saved as conversation history.

Memory persists across separate executions and is isolated by `THREAD_ID`. The
default thread is `default`. Use another thread when you want a separate
conversation:

```sh
# PowerShell
$env:THREAD_ID="conference-planning"
node main.js "What did I previously say about the venue?"
```

The response is shown twice by design: once as the newest assistant entry in
the required full conversation history, and once under `Final response`.

## Optional configuration

Only `OPENROUTER_API_KEY` and `HF_API_KEY` are required. Every other setting has
a default:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MODEL_NAME` | `nvidia/nemotron-3-nano-30b-a3b:free` | OpenRouter reasoning model |
| `CHROMA_HOST` | `localhost` | Chroma server host |
| `CHROMA_PORT` | `8000` | Chroma server port |
| `CHROMA_SSL` | `false` | Use HTTPS for Chroma |
| `AUTO_START_CHROMA` | `true` | Start Chroma automatically for a local host |
| `CHROMA_COMMAND` | `chroma` | Chroma executable name or path |
| `THREAD_ID` | `default` | Conversation-memory namespace |
| `CHUNK_SIZE` | `1000` | Maximum chunk length |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RETRIEVAL_COUNT` | `5` | Maximum number of fused retrieval results |

For a remote Chroma deployment, set its host, port, and SSL options and disable
automatic startup.

## Troubleshooting

### Failed to connect to ChromaDB

Confirm that Chroma is installed and its command is available:

```sh
chroma run --help
```

If it is not available, install it with `pip install chromadb`. When Python's
scripts directory is not on `PATH`, set `CHROMA_COMMAND` to the full path of
the `chroma` executable or start the server manually.

### The agent says no internal information was found

Confirm that readable documents (not only `.gitkeep`) exist under `data/`, then
run the command again. Ask a focused question such as "According to the
internal documents..." so the agent knows to use the retrieval tool.

### Check the JavaScript syntax

```sh
npm run check
```
