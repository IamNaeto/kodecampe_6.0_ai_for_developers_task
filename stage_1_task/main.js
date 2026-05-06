require('dotenv').config({ path: __dirname + '/.env' });
const readline = require('readline');

async function callLlmApi(prompt) {
    const API_KEY = process.env.OPENROUTER_API_KEY;
    const url = "https://openrouter.ai/api/v1/chat/completions";

    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Authorization": `Bearer ${API_KEY}`,
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            model: process.env.MODEL_NAME,
            messages: [{ role: "user", content: prompt }],
            max_tokens: 500
        }),
    });

    const result = await response.json();

    if (result.error) throw new Error(result.error.message);

    return result?.choices?.[0]?.message?.content;
}

async function main() {
    const prompt = process.argv.slice(2).join(" ");

    if (!prompt) {
        console.log("Usage: node main.js \"Your prompt here\"");
        return;
    }

    try {
        const answer = await callLlmApi(prompt);
        console.log("\nAI Response:\n", answer);
    } catch (error) {
        console.error("Error:", error.message);
    }
}

main();