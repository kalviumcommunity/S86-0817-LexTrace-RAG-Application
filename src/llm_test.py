import os
import logging

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError

load_dotenv()

# Basic logging
logging.basicConfig(level=logging.INFO)

base_url = os.getenv("GEMINI_BASE_URL")
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("CHAT_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

if not base_url:
    raise ValueError("GEMINI_BASE_URL is missing from .env")

if not model:
    raise ValueError("CHAT_MODEL is missing from .env")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

messages = [
    {
        "role": "system",
        "content": "You are a concise assistant."
    },
    {
        "role": "user",
        "content": "Say hello in one sentence."
    }
]

logging.info("Sending request to Gemini...")
logging.info("Model: %s", model)

try:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    answer = response.choices[0].message.content

    print("\nLexTrace LLM connection successful!")
    print("Response:", answer)

    logging.info("Response: %s", answer)

    if response.usage:
        logging.info("Usage: %s", response.usage)

except AuthenticationError:
    print("Authentication failed (401): Check your Gemini API key.")

except RateLimitError:
    print("Rate limit/quota error (429): Please try again later.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")