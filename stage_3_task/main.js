import "dotenv/config";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------------------------------
// Environment & config
// ---------------------------------------------------------------------------
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const MODEL_NAME = process.env.MODEL_NAME;

if (!OPENROUTER_API_KEY) {
  console.error("Error: OPENROUTER_API_KEY is not set. Add it to your .env file.");
  process.exit(1);
}
if (!MODEL_NAME) {
  console.error("Error: MODEL_NAME is not set. Add it to your .env file.");
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Read the client project description from the first CLI argument
// ---------------------------------------------------------------------------
const projectDescription = process.argv[2];

if (!projectDescription) {
  console.error(
    'Usage: node main.js "<client project description>"\n' +
      'Example: node main.js "We need a system for our shop to sell items online."'
  );
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function loadPrompt(fileName) {
  const filePath = path.join(__dirname, "prompts", fileName);
  return fs.readFileSync(filePath, "utf-8");
}

function banner(title) {
  const line = "=".repeat(70);
  return `\n${line}\n${title}\n${line}`;
}

// ---------------------------------------------------------------------------
// Model — pointed at OpenRouter's OpenAI-compatible endpoint
// ---------------------------------------------------------------------------
const model = new ChatOpenAI({
  apiKey: OPENROUTER_API_KEY,
  model: MODEL_NAME,
  temperature: 0.3,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  },
});

const parser = new StringOutputParser();

// ---------------------------------------------------------------------------
// Load the five prompt templates
// ---------------------------------------------------------------------------
const interpretPrompt = PromptTemplate.fromTemplate(loadPrompt("01_interpret.txt"));
const categoriesPrompt = PromptTemplate.fromTemplate(loadPrompt("02_categories.txt"));
const selectPrompt = PromptTemplate.fromTemplate(loadPrompt("03_select.txt"));
const missingPrompt = PromptTemplate.fromTemplate(loadPrompt("04_missing.txt"));
const assessmentPrompt = PromptTemplate.fromTemplate(loadPrompt("05_assessment.txt"));

// ---------------------------------------------------------------------------
// Individual LCEL stage chains (prompt | model | parser)
// ---------------------------------------------------------------------------
const interpretChain = interpretPrompt.pipe(model).pipe(parser);
const categoriesChain = categoriesPrompt.pipe(model).pipe(parser);
const selectChain = selectPrompt.pipe(model).pipe(parser);
const missingChain = missingPrompt.pipe(model).pipe(parser);
const assessmentChain = assessmentPrompt.pipe(model).pipe(parser);

// ---------------------------------------------------------------------------
// Full pipeline composed with LCEL.
// Each stage runs, its output is printed, then it is merged into the running
// state object that is passed forward to the next stage.
// ---------------------------------------------------------------------------
const chain = RunnableSequence.from([
  // Stage 1: Interpret the project request
  RunnablePassthrough.assign({
    interpretation: (input) => interpretChain.invoke(input),
  }),
  (state) => {
    console.log(banner("STAGE 1 — PROJECT INTERPRETATION"));
    console.log(state.interpretation.trim());
    return state;
  },

  // Stage 2: Identify possible categories
  RunnablePassthrough.assign({
    candidate_categories: (input) => categoriesChain.invoke(input),
  }),
  (state) => {
    console.log(banner("STAGE 2 — POSSIBLE CATEGORIES"));
    console.log(state.candidate_categories.trim());
    return state;
  },

  // Stage 3: Select the best category
  RunnablePassthrough.assign({
    selected_category: (input) => selectChain.invoke(input),
  }),
  (state) => {
    console.log(banner("STAGE 3 — SELECTED CATEGORY"));
    console.log(state.selected_category.trim());
    return state;
  },

  // Stage 4: Extract missing requirements
  RunnablePassthrough.assign({
    missing_requirements: (input) => missingChain.invoke(input),
  }),
  (state) => {
    console.log(banner("STAGE 4 — MISSING REQUIREMENTS"));
    console.log(state.missing_requirements.trim());
    return state;
  },

  // Stage 5: Generate the initial assessment
  RunnablePassthrough.assign({
    assessment: (input) => assessmentChain.invoke(input),
  }),
  (state) => {
    console.log(banner("STAGE 5 — INITIAL ASSESSMENT"));
    console.log(state.assessment.trim());
    return state;
  },
]);

// ---------------------------------------------------------------------------
// Run it
// ---------------------------------------------------------------------------
async function run() {
  console.log(banner("CLIENT PROJECT DESCRIPTION"));
  console.log(projectDescription);

  const result = await chain.invoke({ project_description: projectDescription });

  console.log(banner("FINAL PROJECT ASSESSMENT"));
  console.log(result.assessment.trim());
}

run().catch((err) => {
  console.error("\nPipeline failed:", err);
  process.exit(1);
});
