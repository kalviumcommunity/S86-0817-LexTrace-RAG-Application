import os
import logging

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError

load_dotenv()

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

# Experiment 1: System role vs User role

messages = [
    {
        "role": "system",
        "content": (
            "You are LexTrace, a concise legal document assistant. "
            "Give clear and factual explanations."
        )
    },
    {
        "role": "user",
        "content": "What is contract termination?"
    }
]

print("\n--- Experiment 1: System + User Roles ---")

try:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    answer = response.choices[0].message.content
    print("Response:", answer)

except AuthenticationError:
    print("Authentication failed (401): Check your Gemini API key.")

except RateLimitError:
    print("Rate limit/quota error (429): Please try again later.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Experiment 2: Compare prompt variations

prompts = [
    "Explain contract termination.",
    "In 2 sentences, explain what contract termination means."
]

print("\n--- Experiment 2: Prompt Comparison ---")

for prompt in prompts:

    print(f"\nPrompt: {prompt}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are LexTrace, a concise and factual "
                        "legal document assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content
        print("Response:", answer)

    except AuthenticationError:
        print("Authentication failed (401): Check your Gemini API key.")

    except RateLimitError:
        print("Rate limit/quota error (429): Please try again later.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")