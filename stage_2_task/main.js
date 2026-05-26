import { config } from "dotenv";
import fs from "fs/promises";
import path from "path";

config();

const BASE_API_URL = "https://openrouter.ai/api/v1/chat/completions";

const CATEGORIES = [
    "Account Opening",
    "Billing Issue",
    "Account Access",
    "Transaction Inquiry",
    "Card Services",
    "Account Statement",
    "Loan Inquiry",
    "General Information"
];

async function loadPrompt(filename) {
    return fs.readFile(path.join("prompts", filename), "utf-8");
}

async function callLlmApi(prompt) {
    const API_KEY = process.env.OPENROUTER_API_KEY;
    const MODEL_NAME = process.env.MODEL_NAME;

    if (!API_KEY) throw new Error("Missing OPENROUTER_API_KEY in .env");
    if (!MODEL_NAME) throw new Error("Missing MODEL_NAME in .env");

    const response = await fetch(BASE_API_URL, {
        method: "POST",
        headers: {
            Authorization: `Bearer ${API_KEY}`,
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            model: MODEL_NAME,
            messages: [{ role: "user", content: prompt }],
            max_tokens: 500
        })
    });

    const result = await response.json();

    if (!response.ok || result.error) {
        throw new Error(result.error?.message || "OpenRouter API request failed");
    }

    return result.choices?.[0]?.message?.content?.trim();
}

function fillTemplate(template, variables) {
    let output = template;

    for (const [key, value] of Object.entries(variables)) {
        output = output.replaceAll(`{{${key}}}`, value);
    }

    return output;
}

async function promptChain(customerQuery) {
    const categoriesText = CATEGORIES.join(", ");

    const prompt1 = fillTemplate(await loadPrompt("01_intent.txt"), {
        CUSTOMER_QUERY: customerQuery
    });

    const step1 = await callLlmApi(prompt1);
    console.log("\nSTEP 1 - Intent:\n", step1);

    const prompt2 = fillTemplate(await loadPrompt("02_possible_categories.txt"), {
        CUSTOMER_QUERY: customerQuery,
        INTENT: step1,
        CATEGORIES: categoriesText
    });

    const step2 = await callLlmApi(prompt2);
    console.log("\nSTEP 2 - Possible Categories:\n", step2);

    const prompt3 = fillTemplate(await loadPrompt("03_choose_category.txt"), {
        CUSTOMER_QUERY: customerQuery,
        INTENT: step1,
        POSSIBLE_CATEGORIES: step2,
        CATEGORIES: categoriesText
    });

    const step3 = await callLlmApi(prompt3);
    console.log("\nSTEP 3 - Chosen Category:\n", step3);

    const prompt4 = fillTemplate(await loadPrompt("04_extract_details.txt"), {
        CUSTOMER_QUERY: customerQuery,
        INTENT: step1,
        CHOSEN_CATEGORY: step3
    });

    const step4 = await callLlmApi(prompt4);
    console.log("\nSTEP 4 - Extracted Details:\n", step4);

    const prompt5 = fillTemplate(await loadPrompt("05_generate_response.txt"), {
        CUSTOMER_QUERY: customerQuery,
        INTENT: step1,
        CHOSEN_CATEGORY: step3,
        EXTRACTED_DETAILS: step4
    });

    const step5 = await callLlmApi(prompt5);
    console.log("\nFINAL RESPONSE:\n", step5);

    return step5;
}

const customerQuery = process.argv.slice(2).join(" ");

if (!customerQuery) {
    console.error('Usage: node main.js "customer query here"');
    process.exit(1);
}

promptChain(customerQuery).catch((error) => {
    console.error("Error:", error.message);
    process.exit(1);
});