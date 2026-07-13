import "dotenv/config";

import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { GoogleGenAI } from "@google/genai";
import { InferenceClient } from "@huggingface/inference";
import { ChromaClient } from "chromadb";
import express from "express";
import mammoth from "mammoth";
import multer from "multer";
import pdfParse from "pdf-parse";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function requiredEnvironment(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

function positiveInteger(name) {
  const value = Number.parseInt(requiredEnvironment(name), 10);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

const config = {
  hfApiKey: requiredEnvironment("HF_API_KEY"),
  embedModel: requiredEnvironment("EMBED_MODEL_NAME"),
  geminiApiKey: requiredEnvironment("GEMINI_API_KEY"),
  llmModel: requiredEnvironment("LLM_MODEL_NAME"),
  chromaHost: requiredEnvironment("CHROMA_DB_HOST"),
  chromaPort: positiveInteger("CHROMA_DB_PORT"),
  dataDir: path.resolve(__dirname, requiredEnvironment("RAG_DATA_DIR")),
  chunkLength: positiveInteger("CHUNK_LENGTH"),
  serverPort: positiveInteger("SERVER_PORT"),
};

const COLLECTION_NAME = "uploaded_documents";
const RETRIEVAL_COUNT = 5;
const EMBEDDING_BATCH_SIZE = 16;

await fs.mkdir(config.dataDir, { recursive: true });

const hf = new InferenceClient(config.hfApiKey);
const gemini = new GoogleGenAI({ apiKey: config.geminiApiKey });

function createChromaClient() {
  let host = config.chromaHost;
  let ssl = false;

  try {
    const url = new URL(config.chromaHost);
    host = url.hostname;
    ssl = url.protocol === "https:";
  } catch {
    // A hostname such as "localhost" is also a valid CHROMA_DB_HOST.
  }

  return new ChromaClient({ host, port: config.chromaPort, ssl });
}

const chroma = createChromaClient();
let collectionPromise;

function getCollection() {
  collectionPromise ??= chroma.getOrCreateCollection({
    name: COLLECTION_NAME,
    metadata: { description: "RAG document chunks" },
    configuration: { hnsw: { space: "cosine" } },
  });
  return collectionPromise.catch((error) => {
    collectionPromise = undefined;
    throw error;
  });
}

function normalizeWhitespace(text) {
  return text
    .replace(/\r\n?/g, "\n")
    .replace(/[\t\f\v ]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// Sentence/paragraph-aware chunks retain semantic boundaries while respecting
// the configurable maximum length. Overlong sentences are split on words.
function semanticChunk(text, maximumLength) {
  const normalized = normalizeWhitespace(text);
  if (!normalized) return [];

  const segments = normalized
    .split(/(?<=[.!?])\s+|\n{2,}/u)
    .map((segment) => segment.trim())
    .filter(Boolean);

  const units = [];
  for (const segment of segments) {
    if (segment.length <= maximumLength) {
      units.push(segment);
      continue;
    }

    const words = segment
      .split(/\s+/u)
      .flatMap((word) =>
        word.length <= maximumLength
          ? [word]
          : Array.from(
              { length: Math.ceil(word.length / maximumLength) },
              (_, index) => word.slice(index * maximumLength, (index + 1) * maximumLength),
            ),
      );
    let unit = "";
    for (const word of words) {
      if (!unit) {
        unit = word;
      } else if (unit.length + word.length + 1 <= maximumLength) {
        unit += ` ${word}`;
      } else {
        units.push(unit);
        unit = word;
      }
    }
    if (unit) units.push(unit);
  }

  const chunks = [];
  let chunk = "";
  for (const unit of units) {
    if (!chunk) {
      chunk = unit;
    } else if (chunk.length + unit.length + 1 <= maximumLength) {
      chunk += ` ${unit}`;
    } else {
      chunks.push(chunk);
      chunk = unit;
    }
  }
  if (chunk) chunks.push(chunk);
  return chunks;
}

async function extractText(file) {
  const extension = path.extname(file.originalname).toLowerCase();
  const buffer = await fs.readFile(file.path);

  if (extension === ".pdf") {
    return (await pdfParse(buffer)).text;
  }
  if (extension === ".docx") {
    return (await mammoth.extractRawText({ buffer })).value;
  }
  if ([".txt", ".md", ".csv", ".json", ".html", ".htm", ".xml"].includes(extension)) {
    return buffer.toString("utf8");
  }

  throw new Error(
    `Unsupported file type '${extension || "unknown"}' for ${file.originalname}`,
  );
}

function normalizeEmbedding(result) {
  if (ArrayBuffer.isView(result)) return Array.from(result);
  if (Array.isArray(result) && result.every((value) => typeof value === "number")) {
    return result;
  }
  if (Array.isArray(result) && result.length === 1) {
    return normalizeEmbedding(result[0]);
  }
  throw new Error("Hugging Face returned an unexpected embedding shape");
}

async function embed(text) {
  const result = await hf.featureExtraction({
    model: config.embedModel,
    inputs: text,
  });
  return normalizeEmbedding(result);
}

async function embedMany(texts) {
  const embeddings = [];
  for (let offset = 0; offset < texts.length; offset += EMBEDDING_BATCH_SIZE) {
    const batch = texts.slice(offset, offset + EMBEDDING_BATCH_SIZE);
    // Individual calls avoid provider-dependent tensor shapes for batched input.
    embeddings.push(...(await Promise.all(batch.map((text) => embed(text)))));
  }
  return embeddings;
}

const storage = multer.diskStorage({
  destination: (_request, _file, callback) => callback(null, config.dataDir),
  filename: (_request, file, callback) => {
    const safeName = path.basename(file.originalname).replace(/[^a-zA-Z0-9._-]/g, "_");
    callback(null, `${Date.now()}-${randomUUID()}-${safeName}`);
  },
});

const upload = multer({
  storage,
  limits: { files: 20, fileSize: 25 * 1024 * 1024 },
});

const app = express();
app.use(express.json({ limit: "1mb" }));

// All required endpoints are intentionally defined in this main file.
app.get("/health", (_request, response) => {
  response.status(200).json({ status: "ok" });
});

app.post("/upload", upload.array("files"), async (request, response, next) => {
  try {
    if (!request.files?.length) {
      return response.status(400).json({ error: "Upload at least one file using the 'files' field" });
    }

    const documents = [];
    const metadatas = [];
    const ids = [];
    const uploaded = [];

    for (const file of request.files) {
      const chunks = semanticChunk(await extractText(file), config.chunkLength);
      if (!chunks.length) throw new Error(`${file.originalname} contains no readable text`);

      chunks.forEach((chunk, index) => {
        documents.push(chunk);
        metadatas.push({ source: file.originalname, chunk: index });
        ids.push(randomUUID());
      });
      uploaded.push({ filename: file.originalname, chunks: chunks.length });
    }

    const collection = await getCollection();
    const embeddings = await embedMany(documents);
    await collection.add({ ids, documents, embeddings, metadatas });

    return response.status(201).json({
      message: "Files uploaded and indexed successfully",
      files: uploaded,
      chunksIndexed: documents.length,
    });
  } catch (error) {
    return next(error);
  }
});

app.post("/prompt", async (request, response, next) => {
  try {
    const query = typeof request.body?.query === "string" ? request.body.query.trim() : "";
    if (!query) return response.status(400).json({ error: "'query' must be a non-empty string" });

    const collection = await getCollection();
    const result = await collection.query({
      queryEmbeddings: [await embed(query)],
      nResults: RETRIEVAL_COUNT,
      include: ["documents", "metadatas", "distances"],
    });

    const documents = result.documents?.[0]?.filter(Boolean) ?? [];
    const metadata = result.metadatas?.[0] ?? [];
    const context = documents
      .map((document, index) => `[Source: ${metadata[index]?.source ?? "unknown"}]\n${document}`)
      .join("\n\n---\n\n");

    if (!context) {
      return response.status(409).json({ error: "No indexed context is available. Upload files first." });
    }

    const llmResponse = await gemini.models.generateContent({
      model: config.llmModel,
      contents: `You are a retrieval-augmented assistant. Answer the question using only the supplied context. If the context does not contain the answer, say that the uploaded documents do not provide enough information.\n\nCONTEXT\n${context}\n\nQUESTION\n${query}`,
      config: { temperature: 0.2 },
    });

    return response.status(200).json({
      answer: llmResponse.text,
      sources: [...new Set(metadata.map((item) => item?.source).filter(Boolean))],
    });
  } catch (error) {
    return next(error);
  }
});

app.use((error, _request, response, _next) => {
  console.error(error);
  if (error instanceof multer.MulterError) {
    return response.status(400).json({ error: error.message });
  }
  return response.status(500).json({ error: error.message || "Internal server error" });
});

app.listen(config.serverPort, () => {
  console.log(`RAG API listening on port ${config.serverPort}`);
});
