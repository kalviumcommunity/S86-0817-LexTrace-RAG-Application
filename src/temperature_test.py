import os

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")
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
        "content": (
            "You are LexTrace, a factual legal document assistant. "
            "Do not invent legal information."
        ),
    },
    {
        "role": "user",
        "content": (
            "Explain why a contract might be terminated. "
            "Give a short answer."
        ),
    },
]


def generate_answer(temperature):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=100,
        )

        return response.choices[0].message.content

    except AuthenticationError:
        return "Authentication failed (401): Check your Gemini API key."

    except RateLimitError:
        return "Rate limit/quota error (429): Please try again later."

    except Exception as e:
        return f"Unexpected error: {e}"


print("\n--- Model Parameter Test ---")

for temperature in [0.0, 1.0]:
    print(f"\nTemperature: {temperature}")

    answer = generate_answer(temperature)

    print("Response:")
    print(answer)