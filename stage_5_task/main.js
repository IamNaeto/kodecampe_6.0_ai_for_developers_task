import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config({ quiet: true });

const USER_PROMPT =
  "I'm taking a flight from Lagos to Nairobi for a conference. I would like to know the total flight time back and forth, and the total cost of logistics for this conference if I'm staying for three days.";

const requiredEnvironmentVariable = (name) => {
  const value = process.env[name]?.trim();

  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }

  return value;
};

/**
 * Return an illustrative round-trip flight booking schedule.
 * In a production system this function could call a live flight provider.
 */
function getFlightBookingSchedule({ origin, destination, trip_type }) {
  if (
    !origin.trim().toLowerCase().includes("lagos") ||
    !destination.trim().toLowerCase().includes("nairobi")
  ) {
    throw new Error("This demo only has flight data for Lagos to Nairobi.");
  }

  if (trip_type !== "round_trip") {
    throw new Error("The conference itinerary requires a round trip.");
  }

  return {
    currency: "USD",
    itinerary: [
      {
        leg: "outbound",
        route: "Lagos (LOS) to Nairobi (NBO)",
        departure: "Conference travel day, 10:00 WAT",
        arrival: "Conference travel day, 17:15 EAT",
        flight_time_minutes: 315,
        price_usd: 620,
      },
      {
        leg: "return",
        route: "Nairobi (NBO) to Lagos (LOS)",
        departure: "Day after the three-night stay, 18:00 EAT",
        arrival: "Same day, 21:15 WAT",
        flight_time_minutes: 315,
        price_usd: 580,
      },
    ],
    total_flight_time_minutes: 630,
    total_price_usd: 1200,
    note: "Illustrative schedule and pricing; no travel dates were supplied.",
  };
}

/**
 * Return an illustrative Nairobi hotel booking schedule priced in USD.
 */
function getHotelBookingSchedule({ city, nights }) {
  if (!city.trim().toLowerCase().includes("nairobi")) {
    throw new Error("This demo only has hotel data for Nairobi.");
  }

  if (!Number.isInteger(nights) || nights <= 0) {
    throw new Error("Hotel nights must be a positive integer.");
  }

  const nightlyRateUsd = 150;

  return {
    city: "Nairobi",
    hotel: "Conference Centre Hotel",
    stay: {
      check_in: "Conference arrival day",
      check_out: `After ${nights} nights`,
      nights,
    },
    currency: "USD",
    nightly_rate_usd: nightlyRateUsd,
    total_price_usd: nightlyRateUsd * nights,
    note: "Illustrative room-only pricing; taxes are included.",
  };
}

/**
 * Convert money using fixed demo rates so the result is deterministic.
 */
function convertCurrency({ amount, from_currency, to_currency }) {
  if (!Number.isFinite(amount) || amount < 0) {
    throw new Error("The amount must be a non-negative number.");
  }

  const from = from_currency.toUpperCase();
  const to = to_currency.toUpperCase();
  const usdRates = {
    USD: 1,
    NGN: 1600,
    KES: 129,
    EUR: 0.92,
    GBP: 0.78,
  };

  if (!usdRates[from] || !usdRates[to]) {
    throw new Error(
      `Unsupported currency. Supported currencies: ${Object.keys(usdRates).join(", ")}.`,
    );
  }

  const convertedAmount = (amount / usdRates[from]) * usdRates[to];

  return {
    original_amount: amount,
    from_currency: from,
    converted_amount: Number(convertedAmount.toFixed(2)),
    to_currency: to,
    exchange_rate: Number((usdRates[to] / usdRates[from]).toFixed(6)),
    rate_note: "Fixed illustrative exchange rate, not a live market quote.",
  };
}

const toolDefinitions = [
  {
    type: "function",
    function: {
      name: "get_flight_booking_schedule",
      description:
        "Get the round-trip flight schedule, flight times, and USD prices for Lagos and Nairobi.",
      parameters: {
        type: "object",
        properties: {
          origin: {
            type: "string",
            description: "The departure city.",
          },
          destination: {
            type: "string",
            description: "The destination city.",
          },
          trip_type: {
            type: "string",
            enum: ["round_trip"],
            description: "The type of flight itinerary.",
          },
        },
        required: ["origin", "destination", "trip_type"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_hotel_booking_schedule",
      description:
        "Get a hotel schedule and USD pricing for a stay in Nairobi.",
      parameters: {
        type: "object",
        properties: {
          city: {
            type: "string",
            description: "The city where the hotel is required.",
          },
          nights: {
            type: "integer",
            minimum: 1,
            description: "The number of hotel nights.",
          },
        },
        required: ["city", "nights"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "convert_currency",
      description:
        "Convert a monetary total from one currency to another using a fixed illustrative exchange rate.",
      parameters: {
        type: "object",
        properties: {
          amount: {
            type: "number",
            minimum: 0,
            description:
              "The total flight and hotel cost obtained from the other tools.",
          },
          from_currency: {
            type: "string",
            enum: ["USD"],
            description: "The currency of the booking prices.",
          },
          to_currency: {
            type: "string",
            enum: ["NGN"],
            description:
              "The currency to convert the logistics total into for the Lagos traveler.",
          },
        },
        required: ["amount", "from_currency", "to_currency"],
        additionalProperties: false,
      },
    },
  },
];

const toolImplementations = {
  get_flight_booking_schedule: getFlightBookingSchedule,
  get_hotel_booking_schedule: getHotelBookingSchedule,
  convert_currency: convertCurrency,
};

const requiredToolOrder = [
  "get_flight_booking_schedule",
  "get_hotel_booking_schedule",
  "convert_currency",
];

async function completeToolConversation(client, model) {
  const messages = [
    {
      role: "system",
      content:
        "You are a travel logistics assistant. Use the supplied tools to answer the user. First obtain the round-trip flight schedule, then the three-night hotel schedule. Add their USD totals and pass that exact combined amount to the currency converter to obtain an NGN equivalent. In the final response, clearly state the outbound, return, and total flight duration; itemize flight and hotel costs; report the total in USD and NGN; and label schedules, prices, and exchange rates as illustrative. Do not invent values that contradict tool results.",
    },
    { role: "user", content: USER_PROMPT },
  ];
  const completedTools = new Set();
  const bookingCostsUsd = {};

  for (let turn = 0; turn < 10; turn += 1) {
    const nextRequiredTool = requiredToolOrder.find(
      (toolName) => !completedTools.has(toolName),
    );
    const completion = await client.chat.completions.create({
      model,
      messages,
      tools: toolDefinitions,
      tool_choice: nextRequiredTool
        ? { type: "function", function: { name: nextRequiredTool } }
        : "none",
    });
    const assistantMessage = completion.choices[0]?.message;

    if (!assistantMessage) {
      throw new Error("The LLM returned no assistant message.");
    }

    messages.push(assistantMessage);

    if (!assistantMessage.tool_calls?.length) {
      if (nextRequiredTool) {
        throw new Error(
          `The LLM did not call the required tool: ${nextRequiredTool}.`,
        );
      }

      if (!assistantMessage.content?.trim()) {
        throw new Error("The LLM returned an empty final response.");
      }

      return assistantMessage.content.trim();
    }

    for (const toolCall of assistantMessage.tool_calls) {
      const toolName = toolCall.function.name;
      const implementation = toolImplementations[toolName];

      if (!implementation) {
        throw new Error(`The LLM requested an unknown tool: ${toolName}.`);
      }

      let result;

      try {
        const argumentsObject = JSON.parse(toolCall.function.arguments);

        if (toolName === "convert_currency") {
          const expectedTotal =
            bookingCostsUsd.flight + bookingCostsUsd.hotel;

          if (
            !Number.isFinite(expectedTotal) ||
            argumentsObject.amount !== expectedTotal
          ) {
            throw new Error(
              `Conversion amount must equal the combined flight and hotel cost of USD ${expectedTotal}.`,
            );
          }
        }

        result = implementation(argumentsObject);

        if (toolName === "get_flight_booking_schedule") {
          bookingCostsUsd.flight = result.total_price_usd;
        } else if (toolName === "get_hotel_booking_schedule") {
          bookingCostsUsd.hotel = result.total_price_usd;
        }

        completedTools.add(toolName);
      } catch (error) {
        result = { error: error.message };
      }

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify(result),
      });
    }
  }

  throw new Error("The tool conversation exceeded the maximum number of turns.");
}

async function main() {
  const apiKey = requiredEnvironmentVariable("OPENROUTER_API_KEY");
  const model = requiredEnvironmentVariable("LLM_MODEL_NAME");
  const client = new OpenAI({
    apiKey,
    baseURL: "https://openrouter.ai/api/v1",
  });

  const finalResponse = await completeToolConversation(client, model);
  console.log(finalResponse);
}

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  process.exitCode = 1;
});
