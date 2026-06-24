# Software Requirements Analysis System

A 5-stage prompt chain built with **LangChain LCEL** (JavaScript) that turns a
client's free-text project description into a structured initial project assessment.

## Reasoning chain

```
Understanding → Classification → Validation → Requirement Extraction → Assessment
```

1. **Interpret the Project Request** — identify the main business objective.
2. **Identify Possible Project Categories** — suggest plausible categories.
3. **Select the Best Category** — choose the single best fit.
4. **Extract Missing Requirements** — list info needed before implementation.
5. **Generate an Initial Assessment** — category, summary, gaps, next steps.

Each stage's output is printed before the next stage runs, and stages are
composed with LCEL (`prompt.pipe(model).pipe(parser)` + `RunnableSequence`).

## Categories

Web Application · Mobile Application · API / Backend Service · Data Analytics
Platform · AI / Machine Learning System · E-Commerce Platform · Enterprise
Management System · System Integration · DevOps / Infrastructure Automation ·
General Software Project

## Setup

```bash
npm install
cp .env.example .env   # then edit .env with your real values
```

`.env`:

```
OPENROUTER_API_KEY=sk-or-...
MODEL_NAME=openai/gpt-4o-mini
```

> The `.env` file is git-ignored and must not be committed.

## Run

Pass the client description as the first command-line argument:

```bash
node main.js "We want a platform where local farmers can list produce and customers can order and pay online."
```

## Project structure

```
.
├── main.js              # LCEL prompt chain
├── prompts/
│   ├── 01_interpret.txt
│   ├── 02_categories.txt
│   ├── 03_select.txt
│   ├── 04_missing.txt
│   └── 05_assessment.txt
├── package.json
├── .env.example
└── .gitignore
```

## Notes

- The model is called through OpenRouter's OpenAI-compatible endpoint via
  `@langchain/openai`'s `ChatOpenAI` (`baseURL` set to
  `https://openrouter.ai/api/v1`).
- To use OpenAI/Anthropic directly instead, set `MODEL_NAME` accordingly and
  remove the `configuration.baseURL` override (or swap to `@langchain/anthropic`).
