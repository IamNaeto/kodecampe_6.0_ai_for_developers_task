# Stage 5: LLM function calling

This program uses the OpenAI SDK with OpenRouter to complete a multi-turn
tool-calling conversation. It supplies three local tools:

- a Lagos–Nairobi round-trip flight schedule with USD pricing;
- a Nairobi hotel schedule with USD pricing;
- currency conversion of the combined logistics cost from USD to NGN.

The schedules, prices, and exchange rate are deterministic illustrative data,
not live booking quotes.

## Setup

1. Install Node.js 20 or newer.
2. Copy `.env.example` to `.env`.
3. Set `OPENROUTER_API_KEY` and `LLM_MODEL_NAME` in `.env`. Choose an
   OpenRouter model that supports tool/function calling.
4. Install and run:

   ```sh
   npm install
   node main.js
   ```

The program makes the complete tool-calling conversation and prints only the
LLM's final answer to standard output. Errors are written to standard error.
