import os
import json

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


SYSTEM_PROMPT = """
You are LexTrace, a legal document assistant.

Return ONLY a valid JSON object in this exact format:

{
  "answer": "string",
  "source": "string"
}

Do not include markdown, explanations, or any text outside the JSON object.
"""


def parse_response(raw):
    """Parse and validate the model's JSON response."""

    try:
        data = json.loads(raw)

    except json.JSONDecodeError:
        return None, "Malformed JSON"

    required_fields = ["answer", "source"]

    missing = [
        field for field in required_fields
        if field not in data
    ]

    if missing:
        return None, f"Missing fields: {missing}"

    return data, None


print("\n--- LexTrace Structured Output Test ---")

try:

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "What are common conditions under which "
                    "a contract may be terminated?"
                )
            }
        ],
        temperature=0,
    )

    raw_response = response.choices[0].message.content

    print("\nRaw response:")
    print(raw_response)

    data, error = parse_response(raw_response)

    if error:
        print("\nJSON validation failed:", error)
    else:
        print("\nJSON validation successful!")
        print("Answer:", data["answer"])
        print("Source:", data["source"])

except AuthenticationError:
    print("Authentication failed (401): Check your Gemini API key.")

except RateLimitError:
    print("Rate limit/quota error (429): Please try again later.")

except Exception as e:
    print(f"Unexpected error: {e}")