# Stage 7: Mastra CLI assistant

A JavaScript Mastra project with a production-style agent, four tools, local RAG,
persistent conversation memory, streamed terminal output, friendly errors, and clean
Ctrl+C shutdown. No web/backend framework is used; `mastra dev` provides the Mastra
server and Studio.

## Requirements

- Node.js 22.13 or newer
- An OpenRouter API key
- An OpenRouter model that supports tool calling

## Setup

```sh
npm install
copy .env-example .env
```

Edit `.env` and set both required variables:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key_here
MODEL_NAME=openai/gpt-4o-mini
```

All API keys must remain in environment variables. `.env` is git-ignored.

## Run

Start the interactive CLI:

```sh
npm run cli
```

Start the Mastra development server and Studio in a separate terminal:

```sh
npm run dev
```

Use `/exit`, `/quit`, or Ctrl+C to exit. By default, a new random thread is made
for each CLI process. Set `MASTRA_THREAD_ID` to a stable value to resume the same
conversation after restarting the CLI.

## Tools and RAG

The registered agent has the same four tools as Stage 6:

1. `get_flight_booking_schedule`
2. `get_hotel_booking_schedule`
3. `convert_currency`
4. `query_internal_information`

The RAG tool chunks supported text files under `data/`, ranks passages using BM25,
and returns source-labelled context to the agent. Add TXT, Markdown, MDX, CSV,
JSON, HTML, XML, JavaScript, or TypeScript documents and restart the process to
refresh the in-memory index. Conversation history is independently persisted by
Mastra Memory in `mastra.db`.

Travel schedules, prices, and conversion rates are illustrative, not live quotes.
