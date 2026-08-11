import "dotenv/config";

import { Mastra } from "@mastra/core/mastra";
import { LibSQLStore } from "@mastra/libsql";

import { assistantAgent } from "./agents/assistant-agent.js";

export const mastra = new Mastra({
  agents: { assistantAgent },
  storage: new LibSQLStore({
    id: "stage-7-mastra-storage",
    url: "file:./mastra.db",
  }),
});
