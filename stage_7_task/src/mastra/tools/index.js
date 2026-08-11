import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createTool } from "@mastra/core/tools";
import { z } from "zod";

const currentDirectory = path.dirname(fileURLToPath(import.meta.url));
const dataDirectory = path.resolve(currentDirectory, "../../../data");
const supportedExtensions = new Set([
  ".txt", ".md", ".mdx", ".csv", ".json", ".html", ".xml", ".js", ".ts",
]);
let cachedChunks;

function words(value) {
  return value.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
}

function chunkText(text, source, size = 1000, overlap = 150) {
  const chunks = [];
  for (let start = 0; start < text.length; start += size - overlap) {
    const content = text.slice(start, start + size).trim();
    if (content) chunks.push({ content, source, tokens: words(content) });
  }
  return chunks;
}

async function listFiles(directory) {
  let entries;
  try {
    entries = await fs.readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const files = await Promise.all(entries
    .filter((entry) => !entry.name.startsWith("."))
    .map((entry) => {
      const entryPath = path.join(directory, entry.name);
      return entry.isDirectory() ? listFiles(entryPath) : [entryPath];
    }));
  return files.flat().sort();
}

async function loadChunks() {
  if (cachedChunks) return cachedChunks;
  const chunks = [];
  for (const filePath of await listFiles(dataDirectory)) {
    if (!supportedExtensions.has(path.extname(filePath).toLowerCase())) continue;
    const text = await fs.readFile(filePath, "utf8");
    if (text.includes("\u0000")) continue;
    const source = path.relative(dataDirectory, filePath).replaceAll("\\", "/");
    chunks.push(...chunkText(text, source));
  }
  cachedChunks = chunks;
  return chunks;
}

function bm25(query, documents) {
  const queryTerms = [...new Set(words(query))];
  if (!queryTerms.length || !documents.length) return [];
  const averageLength = documents.reduce((sum, item) => sum + item.tokens.length, 0) / documents.length || 1;

  return documents.map((document) => {
    const frequency = new Map();
    for (const token of document.tokens) frequency.set(token, (frequency.get(token) ?? 0) + 1);
    let score = 0;
    for (const term of queryTerms) {
      const documentFrequency = documents.filter((item) => item.tokens.includes(term)).length;
      const inverseDocumentFrequency = Math.log(1 + (documents.length - documentFrequency + 0.5) / (documentFrequency + 0.5));
      const termFrequency = frequency.get(term) ?? 0;
      const denominator = termFrequency + 1.5 * (0.25 + 0.75 * document.tokens.length / averageLength);
      score += inverseDocumentFrequency * (termFrequency * 2.5 / denominator);
    }
    return { ...document, score };
  }).filter((item) => item.score > 0).sort((a, b) => b.score - a.score);
}

export const getFlightBookingSchedule = createTool({
  id: "get_flight_booking_schedule",
  description: "Get illustrative Lagos-to-Nairobi round-trip flight times and USD prices.",
  inputSchema: z.object({
    origin: z.string().describe("Departure city"),
    destination: z.string().describe("Destination city"),
    trip_type: z.literal("round_trip"),
  }),
  execute: async ({ origin, destination, trip_type }) => {
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
      note: "Illustrative schedule and pricing, not a live quote.",
    };
  },
});

export const getHotelBookingSchedule = createTool({
  id: "get_hotel_booking_schedule",
  description: "Get an illustrative Nairobi hotel schedule and USD price.",
  inputSchema: z.object({ city: z.string(), nights: z.number().int().positive() }),
  execute: async ({ city, nights }) => {
    if (!city.toLowerCase().includes("nairobi")) {
      throw new Error("Illustrative hotel data is only available in Nairobi.");
    }
    return {
      city: "Nairobi",
      hotel: "Conference Centre Hotel",
      nights,
      currency: "USD",
      nightly_rate_usd: 150,
      total_price_usd: 150 * nights,
      note: "Illustrative room-only pricing with taxes included, not a live quote.",
    };
  },
});

export const convertCurrency = createTool({
  id: "convert_currency",
  description: "Convert an amount using fixed illustrative currency rates.",
  inputSchema: z.object({
    amount: z.number().nonnegative(),
    from_currency: z.enum(["USD", "NGN", "KES", "EUR", "GBP"]),
    to_currency: z.enum(["USD", "NGN", "KES", "EUR", "GBP"]),
  }),
  execute: async ({ amount, from_currency, to_currency }) => {
    const ratesPerUsd = { USD: 1, NGN: 1600, KES: 129, EUR: 0.92, GBP: 0.78 };
    return {
      original_amount: amount,
      from_currency,
      converted_amount: Number((amount / ratesPerUsd[from_currency] * ratesPerUsd[to_currency]).toFixed(2)),
      to_currency,
      rate_note: "Fixed illustrative exchange rate, not a live quote.",
    };
  },
});

export const queryInternalInformation = createTool({
  id: "query_internal_information",
  description: "Retrieve relevant passages from internal files in data/. Use this before answering questions about internal knowledge.",
  inputSchema: z.object({ query: z.string().min(1).describe("A focused internal-search query") }),
  execute: async ({ query }) => {
    const results = bm25(query, await loadChunks()).slice(0, 5);
    if (!results.length) return { found: false, message: "No relevant internal documents were found." };
    return {
      found: true,
      passages: results.map(({ source, content }, index) => ({ rank: index + 1, source: `data/${source}`, content })),
    };
  },
});

export const assistantTools = {
  get_flight_booking_schedule: getFlightBookingSchedule,
  get_hotel_booking_schedule: getHotelBookingSchedule,
  convert_currency: convertCurrency,
  query_internal_information: queryInternalInformation,
};
