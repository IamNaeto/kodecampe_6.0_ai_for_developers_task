import dotenv from "dotenv";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Embeddings } from "@langchain/core/embeddings";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChromaClient } from "chromadb";
import { z } from "zod";

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2";
const DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free";
const KNOWLEDGE_COLLECTION = "stage_6_internal_knowledge";
const MEMORY_COLLECTION = "stage_6_conversation_memory";

function requiredEnvironment(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

function positiveInteger(value, fallback) {
  const parsed = Number.parseInt(value ?? "", 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

const config = {
  openRouterApiKey: requiredEnvironment("OPENROUTER_API_KEY"),
  hfApiKey: requiredEnvironment("HF_API_KEY"),
  model: process.env.LLM_MODEL_NAME?.trim() || DEFAULT_MODEL,
  chromaHost: process.env.CHROMA_HOST?.trim() || "localhost",
  chromaPort: positiveInteger(process.env.CHROMA_PORT, 8000),
  chromaSsl: process.env.CHROMA_SSL?.toLowerCase() === "true",
  threadId: process.env.THREAD_ID?.trim() || "default",
  chunkSize: positiveInteger(process.env.CHUNK_SIZE, 1000),
  chunkOverlap: positiveInteger(process.env.CHUNK_OVERLAP, 200),
  retrievalCount: positiveInteger(process.env.RETRIEVAL_COUNT, 5),
  dataDirectory: path.join(__dirname, "data"),
  chromaDataDirectory: path.join(__dirname, ".chroma"),
  autoStartChroma: process.env.AUTO_START_CHROMA?.toLowerCase() !== "false",
  chromaCommand: process.env.CHROMA_COMMAND?.trim() || "chroma",
};

if (config.chunkOverlap >= config.chunkSize) {
  config.chunkOverlap = Math.floor(config.chunkSize / 5);
}

function averageVectors(vectors) {
  const width = vectors[0].length;
  const result = Array(width).fill(0);
  for (const vector of vectors) {
    if (vector.length !== width) throw new Error("Inconsistent embedding dimensions");
    vector.forEach((value, index) => {
      result[index] += value;
    });
  }
  return result.map((value) => value / vectors.length);
}

function normalizeEmbedding(value) {
  if (ArrayBuffer.isView(value)) return Array.from(value);
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("Hugging Face returned an empty or invalid embedding");
  }
  if (value.every((item) => typeof item === "number")) return value;
  if (value.length === 1) return normalizeEmbedding(value[0]);
  const vectors = value.map(normalizeEmbedding);
  return averageVectors(vectors);
}

/** LangChain embedding implementation backed by Hugging Face Inference. */
class HuggingFaceMiniLMEmbeddings extends Embeddings {
  constructor(apiKey) {
    super({ maxConcurrency: 4, maxRetries: 2 });
    this.apiKey = apiKey;
  }

  async embedOne(text) {
    const response = await fetch(
      `https://router.huggingface.co/hf-inference/models/${EMBEDDING_MODEL}/pipeline/feature-extraction`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          inputs: text,
          options: { wait_for_model: true },
        }),
      },
    );

    if (!response.ok) {
      throw new Error(`Hugging Face embedding request failed (${response.status}): ${await response.text()}`);
    }
    return normalizeEmbedding(await response.json());
  }

  async embedDocuments(documents) {
    const results = [];
    for (let index = 0; index < documents.length; index += 8) {
      results.push(...(await Promise.all(documents.slice(index, index + 8).map((text) => this.embedOne(text)))));
    }
    return results;
  }

  embedQuery(document) {
    return this.embedOne(document);
  }
}

const embeddings = new HuggingFaceMiniLMEmbeddings(config.hfApiKey);
const chroma = new ChromaClient({
  host: config.chromaHost,
  port: config.chromaPort,
  ssl: config.chromaSsl,
});

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function isLocalChromaHost(host) {
  return ["localhost", "127.0.0.1", "::1"].includes(host.toLowerCase());
}

async function ensureChromaServer() {
  try {
    await chroma.heartbeat();
    return;
  } catch (connectionError) {
    if (!config.autoStartChroma || !isLocalChromaHost(config.chromaHost)) {
      throw new Error(
        `Failed to connect to Chroma at ${config.chromaHost}:${config.chromaPort}. ` +
        "Start the server or enable AUTO_START_CHROMA for a local host.",
        { cause: connectionError },
      );
    }
  }

  await fs.mkdir(config.chromaDataDirectory, { recursive: true });
  let server;
  try {
    server = spawn(
      config.chromaCommand,
      [
        "run",
        "--path",
        config.chromaDataDirectory,
        "--host",
        config.chromaHost,
        "--port",
        String(config.chromaPort),
      ],
      { detached: true, stdio: "ignore", windowsHide: true },
    );
    server.unref();
  } catch (error) {
    throw new Error(
      "Chroma is not running and could not be started. Install it with " +
      "`pip install chromadb`, or start `chroma run` yourself.",
      { cause: error },
    );
  }

  let startupError;
  server.once("error", (error) => {
    startupError = error;
  });
  for (let attempt = 0; attempt < 40; attempt += 1) {
    await delay(500);
    if (startupError) {
      throw new Error(
        "Chroma is not running and the `chroma` command could not be launched. " +
        "Install it with `pip install chromadb`, or set CHROMA_COMMAND to its executable path.",
        { cause: startupError },
      );
    }
    try {
      await chroma.heartbeat();
      return;
    } catch {
      // The local server can take several seconds to initialize.
    }
  }
  throw new Error(
    "Timed out while starting Chroma. Run `chroma run --path .chroma --port " +
    `${config.chromaPort}` + "` in another terminal to see its startup error.",
  );
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

async function listFiles(directory) {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries
      .filter((entry) => !entry.name.startsWith("."))
      .map((entry) => {
        const itemPath = path.join(directory, entry.name);
        return entry.isDirectory() ? listFiles(itemPath) : [itemPath];
      }),
  );
  return nested.flat().sort();
}

async function extractText(filePath) {
  const extension = path.extname(filePath).toLowerCase();
  const buffer = await fs.readFile(filePath);

  if (extension === ".pdf") {
    const { default: pdfParse } = await import("pdf-parse");
    return (await pdfParse(buffer)).text;
  }
  if (extension === ".docx") {
    const { default: mammoth } = await import("mammoth");
    return (await mammoth.extractRawText({ buffer })).value;
  }

  // All other files are treated as UTF-8 text, allowing new text-based file
  // extensions to be indexed without changing this script.
  const text = buffer.toString("utf8");
  return text.includes("\u0000") ? "" : text;
}

async function getCollections() {
  const configuration = { hnsw: { space: "cosine" } };
  const [knowledge, memory] = await Promise.all([
    chroma.getOrCreateCollection({
      name: KNOWLEDGE_COLLECTION,
      metadata: { description: "Files loaded from stage_6_task/data" },
      configuration,
      embeddingFunction: null,
    }),
    chroma.getOrCreateCollection({
      name: MEMORY_COLLECTION,
      metadata: { description: "User and assistant conversation messages only" },
      configuration,
      embeddingFunction: null,
    }),
  ]);
  return { knowledge, memory };
}

async function indexData(collection) {
  await fs.mkdir(config.dataDirectory, { recursive: true });
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: config.chunkSize,
    chunkOverlap: config.chunkOverlap,
  });
  const chunks = [];

  for (const filePath of await listFiles(config.dataDirectory)) {
    const text = (await extractText(filePath)).trim();
    if (!text) continue;
    const source = path.relative(config.dataDirectory, filePath).replaceAll("\\", "/");
    const documents = await splitter.createDocuments([text], [{ source }]);
    documents.forEach((document, chunk) => {
      chunks.push({
        id: digest(`${source}:${chunk}:${document.pageContent}`),
        text: document.pageContent,
        metadata: { source, chunk, kind: "internal_document" },
      });
    });
  }

  // Rebuild only the internal-data collection so removed/edited files cannot
  // leave stale chunks. Conversation memory is held in a separate collection.
  const existing = await collection.get();
  if (existing.ids.length) await collection.delete({ ids: existing.ids });
  for (let offset = 0; offset < chunks.length; offset += 32) {
    const batch = chunks.slice(offset, offset + 32);
    await collection.upsert({
      ids: batch.map((item) => item.id),
      documents: batch.map((item) => item.text),
      metadatas: batch.map((item) => item.metadata),
      embeddings: await embeddings.embedDocuments(batch.map((item) => item.text)),
    });
  }
  return chunks.length;
}

function tokenize(text) {
  return text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
}

function bm25Rank(query, records) {
  const queryTerms = [...new Set(tokenize(query))];
  if (!queryTerms.length || !records.length) return [];
  const tokenized = records.map((record) => tokenize(record.text));
  const averageLength = tokenized.reduce((total, words) => total + words.length, 0) / tokenized.length || 1;

  return records
    .map((record, index) => {
      const words = tokenized[index];
      const frequencies = new Map();
      words.forEach((word) => frequencies.set(word, (frequencies.get(word) ?? 0) + 1));
      let score = 0;
      for (const term of queryTerms) {
        const documentFrequency = tokenized.filter((tokens) => tokens.includes(term)).length;
        const inverseDocumentFrequency = Math.log(1 + (records.length - documentFrequency + 0.5) / (documentFrequency + 0.5));
        const frequency = frequencies.get(term) ?? 0;
        score += inverseDocumentFrequency * ((frequency * 2.5) / (frequency + 1.5 * (1 - 0.75 + 0.75 * (words.length / averageLength))));
      }
      return { ...record, score };
    })
    .filter((record) => record.score > 0)
    .sort((left, right) => right.score - left.score);
}

async function collectionRecords(collection, namespace, where) {
  const result = await collection.get({ where, include: ["documents", "metadatas"] });
  return result.ids.map((id, index) => ({
    key: `${namespace}:${id}`,
    id,
    namespace,
    text: result.documents?.[index] ?? "",
    metadata: result.metadatas?.[index] ?? {},
  })).filter((record) => record.text);
}

async function hybridSearch(query, collections) {
  const queryEmbedding = await embeddings.embedQuery(query);
  const entries = [
    { name: "knowledge", collection: collections.knowledge, where: undefined },
    { name: "memory", collection: collections.memory, where: { thread_id: config.threadId } },
  ];
  const recordsByCollection = await Promise.all(
    entries.map(({ name, collection, where }) => collectionRecords(collection, name, where)),
  );
  const corpus = recordsByCollection.flat();

  const semanticGroups = await Promise.all(entries.map(async ({ name, collection, where }, index) => {
    if (!recordsByCollection[index].length) return [];
    const result = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: Math.min(config.retrievalCount * 2, recordsByCollection[index].length),
      where,
      include: ["documents", "metadatas", "distances"],
    });
    return (result.ids?.[0] ?? []).map((id, itemIndex) => ({
      key: `${name}:${id}`,
      id,
      namespace: name,
      text: result.documents?.[0]?.[itemIndex] ?? "",
      metadata: result.metadatas?.[0]?.[itemIndex] ?? {},
      distance: result.distances?.[0]?.[itemIndex] ?? 1,
    }));
  }));
  const semantic = semanticGroups.flat().sort((a, b) => a.distance - b.distance);
  const lexical = bm25Rank(query, corpus);
  const fused = new Map();

  for (const ranking of [semantic, lexical]) {
    ranking.forEach((record, rank) => {
      const current = fused.get(record.key) ?? { ...record, fusedScore: 0 };
      current.fusedScore += 1 / (60 + rank + 1);
      fused.set(record.key, current);
    });
  }
  return [...fused.values()]
    .sort((a, b) => b.fusedScore - a.fusedScore)
    .slice(0, config.retrievalCount);
}

function getFlightBookingSchedule({ origin, destination, trip_type }) {
  if (!origin.toLowerCase().includes("lagos") || !destination.toLowerCase().includes("nairobi")) {
    throw new Error("Illustrative flight data is only available from Lagos to Nairobi.");
  }
  if (trip_type !== "round_trip") throw new Error("Only round-trip schedules are available.");
  return {
    currency: "USD",
    itinerary: [
      { leg: "outbound", route: "Lagos (LOS) to Nairobi (NBO)", departure: "10:00 WAT", arrival: "17:15 EAT", flight_time_minutes: 315, price_usd: 620 },
      { leg: "return", route: "Nairobi (NBO) to Lagos (LOS)", departure: "18:00 EAT", arrival: "21:15 WAT", flight_time_minutes: 315, price_usd: 580 },
    ],
    total_flight_time_minutes: 630,
    total_price_usd: 1200,
    note: "Illustrative schedule and pricing.",
  };
}

function getHotelBookingSchedule({ city, nights }) {
  if (!city.toLowerCase().includes("nairobi")) throw new Error("Illustrative hotel data is only available in Nairobi.");
  const nightlyRateUsd = 150;
  return {
    city: "Nairobi",
    hotel: "Conference Centre Hotel",
    nights,
    currency: "USD",
    nightly_rate_usd: nightlyRateUsd,
    total_price_usd: nightlyRateUsd * nights,
    note: "Illustrative room-only pricing with taxes included.",
  };
}

function convertCurrency({ amount, from_currency, to_currency }) {
  const ratesPerUsd = { USD: 1, NGN: 1600, KES: 129, EUR: 0.92, GBP: 0.78 };
  const from = from_currency.toUpperCase();
  const to = to_currency.toUpperCase();
  if (!ratesPerUsd[from] || !ratesPerUsd[to]) throw new Error("Supported currencies: USD, NGN, KES, EUR, GBP.");
  return {
    original_amount: amount,
    from_currency: from,
    converted_amount: Number(((amount / ratesPerUsd[from]) * ratesPerUsd[to]).toFixed(2)),
    to_currency: to,
    rate_note: "Fixed illustrative exchange rate, not a live quote.",
  };
}

function createTools(collections) {
  const flightTool = tool(async (input) => JSON.stringify(getFlightBookingSchedule(input)), {
    name: "get_flight_booking_schedule",
    description: "Get illustrative Lagos-to-Nairobi round-trip flight times and USD prices.",
    schema: z.object({
      origin: z.string().describe("Departure city"),
      destination: z.string().describe("Destination city"),
      trip_type: z.literal("round_trip"),
    }),
  });
  const hotelTool = tool(async (input) => JSON.stringify(getHotelBookingSchedule(input)), {
    name: "get_hotel_booking_schedule",
    description: "Get an illustrative Nairobi hotel schedule and USD price.",
    schema: z.object({ city: z.string(), nights: z.number().int().positive() }),
  });
  const currencyTool = tool(async (input) => JSON.stringify(convertCurrency(input)), {
    name: "convert_currency",
    description: "Convert an amount using fixed illustrative currency rates.",
    schema: z.object({
      amount: z.number().nonnegative(),
      from_currency: z.enum(["USD", "NGN", "KES", "EUR", "GBP"]),
      to_currency: z.enum(["USD", "NGN", "KES", "EUR", "GBP"]),
    }),
  });
  const ragTool = tool(async ({ query }) => {
    const results = await hybridSearch(query, collections);
    if (!results.length) return "No relevant internal documents or conversation memories were found.";
    return results.map((result, index) => {
      const source = result.namespace === "memory"
        ? `conversation memory, turn ${result.metadata.turn ?? "unknown"}`
        : `data/${result.metadata.source ?? "unknown"}`;
      return `[${index + 1}] Source: ${source}\n${result.text}`;
    }).join("\n\n");
  }, {
    name: "query_internal_information",
    description: "Hybrid semantic and BM25 search over files in data/ and past conversation memory. Use for internal facts or recalled user context.",
    schema: z.object({ query: z.string().min(1).describe("A focused search query") }),
  });
  return [flightTool, hotelTool, currencyTool, ragTool];
}

async function loadConversationHistory(memoryCollection) {
  const records = await memoryCollection.get({
    where: { thread_id: config.threadId },
    include: ["documents", "metadatas"],
  });
  return records.ids.map((id, index) => ({
    id,
    text: records.documents?.[index] ?? "",
    metadata: records.metadatas?.[index] ?? {},
  })).sort((a, b) => (a.metadata.sequence ?? 0) - (b.metadata.sequence ?? 0));
}

async function saveConversationTurn(memoryCollection, previousHistory, prompt, response) {
  const previousTurn = previousHistory.reduce((maximum, item) => Math.max(maximum, Number(item.metadata.turn) || 0), 0);
  const turn = previousTurn + 1;
  const timestamp = new Date().toISOString();
  const messages = [
    { role: "user", text: prompt, sequence: turn * 2 - 1 },
    { role: "assistant", text: response, sequence: turn * 2 },
  ];
  await memoryCollection.upsert({
    ids: messages.map((message) => digest(`${config.threadId}:${turn}:${message.role}`)),
    documents: messages.map((message) => `${message.role === "user" ? "User" : "Assistant"}: ${message.text}`),
    metadatas: messages.map((message) => ({
      thread_id: config.threadId,
      turn,
      role: message.role,
      sequence: message.sequence,
      timestamp,
      kind: "conversation_message",
    })),
    embeddings: await embeddings.embedDocuments(messages.map((message) => message.text)),
  });
  return messages;
}

function contentToText(content) {
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) {
    return content.map((part) => typeof part === "string" ? part : part?.text ?? "").join("\n").trim();
  }
  return String(content ?? "").trim();
}

async function main() {
  const prompt = process.argv.slice(2).join(" ").trim();
  if (!prompt) throw new Error('Provide a prompt, for example: node main.js "What is the weather in Jos?"');

  await ensureChromaServer();
  const collections = await getCollections();
  await indexData(collections.knowledge);
  const previousHistory = await loadConversationHistory(collections.memory);
  const conversationMessages = previousHistory.map((item) =>
    item.metadata.role === "assistant"
      ? new AIMessage(item.text.replace(/^Assistant:\s*/i, ""))
      : new HumanMessage(item.text.replace(/^User:\s*/i, "")),
  );

  const llm = new ChatOpenAI({
    apiKey: config.openRouterApiKey,
    model: config.model,
    temperature: 0.2,
    configuration: { baseURL: "https://openrouter.ai/api/v1" },
  });
  const checkpointer = new MemorySaver();
  const agent = createReactAgent({
    llm,
    tools: createTools({ knowledge: collections.knowledge, memory: collections.memory }),
    checkpointSaver: checkpointer,
    prompt: "You are a helpful tool-using assistant. Use query_internal_information for internal documents or relevant past memories. Use the travel tools when applicable. Clearly label illustrative travel prices and exchange rates. Do not expose tool-call traces in the answer.",
  });
  const result = await agent.invoke(
    { messages: [...conversationMessages, new HumanMessage(prompt)] },
    { configurable: { thread_id: config.threadId } },
  );
  const finalResponse = contentToText(result.messages.at(-1)?.content);
  if (!finalResponse) throw new Error("The agent returned an empty response");

  const currentMessages = await saveConversationTurn(collections.memory, previousHistory, prompt, finalResponse);
  const completeHistory = [
    ...previousHistory.map((item) => ({ role: item.metadata.role, text: item.text.replace(/^(User|Assistant):\s*/i, "") })),
    ...currentMessages,
  ];

  console.log("Conversation history:");
  completeHistory.forEach((message) => {
    console.log(`${message.role === "assistant" ? "Assistant" : "User"}: ${message.text}`);
  });
  console.log("\nFinal response:");
  console.log(finalResponse);
}

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  process.exitCode = 1;
});
