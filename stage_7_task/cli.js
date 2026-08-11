#!/usr/bin/env node
import "dotenv/config";

import { randomUUID } from "node:crypto";
import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";

const resourceId = process.env.MASTRA_RESOURCE_ID?.trim() || "local-cli-user";
const threadId = process.env.MASTRA_THREAD_ID?.trim() || `cli-${randomUUID()}`;
const terminal = createInterface({ input: stdin, output: stdout });
let mastra;
let activeController;
let closing = false;

function startSpinner() {
  if (!stdout.isTTY) return () => {};
  const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  let frame = 0;
  let stopped = false;
  const timer = setInterval(() => {
    stdout.write(`\r${frames[frame++ % frames.length]} Thinking…`);
  }, 80);
  return () => {
    if (stopped) return;
    stopped = true;
    clearInterval(timer);
    stdout.write("\r\x1b[2K");
  };
}

function friendlyError(error) {
  const message = error instanceof Error ? error.message : String(error);
  if (/401|unauthorized|api.?key/i.test(message)) return "OpenRouter rejected the API key. Check OPENROUTER_API_KEY in .env.";
  if (/429|rate.?limit/i.test(message)) return "The model is rate-limited. Wait briefly and try again.";
  if (/fetch|network|ECONN|ENOTFOUND|timeout/i.test(message)) return "The model service could not be reached. Check your connection and try again.";
  return message || "An unexpected error occurred.";
}

async function chat(prompt) {
  activeController = new AbortController();
  const stopSpinner = startSpinner();
  let beganWriting = false;
  try {
    const agent = mastra.getAgent("assistantAgent");
    const response = await agent.stream(prompt, {
      memory: { resource: resourceId, thread: threadId },
      abortSignal: activeController.signal,
    });

    for await (const delta of response.textStream) {
      if (!beganWriting) {
        stopSpinner();
        stdout.write("Assistant: ");
        beganWriting = true;
      }
      stdout.write(delta);
    }
    stopSpinner();
    stdout.write(beganWriting ? "\n\n" : "Assistant returned no text.\n\n");
  } catch (error) {
    stopSpinner();
    if (activeController.signal.aborted) stdout.write("\nResponse cancelled.\n");
    else process.stderr.write(`\nError: ${friendlyError(error)}\n\n`);
  } finally {
    activeController = undefined;
  }
}

async function shutdown() {
  if (closing) return;
  closing = true;
  activeController?.abort();
  terminal.close();
  stdout.write("\nGoodbye!\n");
}

process.once("SIGINT", shutdown);
process.once("SIGTERM", shutdown);

async function main() {
  ({ mastra } = await import("./src/mastra/index.js"));
  stdout.write("Mastra CLI Assistant\n");
  stdout.write(`Conversation: ${threadId}\n`);
  stdout.write("Type /exit to quit. Your conversation is remembered for this thread.\n\n");

  while (!closing) {
    let input;
    try {
      input = (await terminal.question("You: ")).trim();
    } catch (error) {
      if (closing || error?.code === "ABORT_ERR") break;
      throw error;
    }
    if (!input) continue;
    if (["/exit", "/quit", "exit", "quit"].includes(input.toLowerCase())) break;
    await chat(input);
  }
  if (!closing) await shutdown();
}

main().catch((error) => {
  process.stderr.write(`Fatal error: ${friendlyError(error)}\n`);
  process.exitCode = 1;
  terminal.close();
});
