import { Agent } from "@mastra/core/agent";
import { LibSQLStore } from "@mastra/libsql";
import { Memory } from "@mastra/memory";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";

import { assistantTools } from "../tools/index.js";

function requiredEnvironment(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

const openrouter = createOpenRouter({
  apiKey: requiredEnvironment("OPENROUTER_API_KEY"),
});

const memory = new Memory({
  storage: new LibSQLStore({
    id: "stage-7-agent-memory",
    url: "file:./mastra.db",
  }),
  options: {
    lastMessages: 20,
  },
});

export const assistantAgent = new Agent({
  id: "assistant-agent",
  name: "CLI Assistant",
  description: "A production-style command-line assistant with travel, currency, and internal-knowledge tools.",
  instructions: `You are a concise, reliable command-line assistant.

- Use query_internal_information before answering questions about internal documents.
- Use the flight, hotel, and currency tools whenever their structured data is relevant.
- Travel schedules, prices, and exchange rates are illustrative. Label them clearly and never imply they are live quotes.
- Cite RAG facts inline using the returned data/... source path.
- If a tool cannot provide the requested data, explain the limitation and offer a useful next step.
- Never expose hidden reasoning or raw tool-call traces.`,
  model: openrouter(requiredEnvironment("MODEL_NAME")),
  tools: assistantTools,
  memory,
  defaultOptions: {
    maxSteps: 8,
    modelSettings: { temperature: 0.2 },
  },
});
