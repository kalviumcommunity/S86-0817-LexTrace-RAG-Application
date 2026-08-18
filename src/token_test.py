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

# 1. Token estimation using the API response

legal_text = """
What are the conditions for terminating this agreement
if either party breaches its confidentiality obligations?
"""

print("\n--- Token / Usage Test ---")
print("Text:", legal_text.strip())


try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a concise legal document assistant."
            },
            {
                "role": "user",
                "content": legal_text
            }
        ],
    )

    answer = response.choices[0].message.content

    print("\nResponse:")
    print(answer)

    # Actual token usage returned by the API
    if response.usage:
        print("\nActual token usage:")
        print("Input tokens:", response.usage.prompt_tokens)
        print("Output tokens:", response.usage.completion_tokens)
        print("Total tokens:", response.usage.total_tokens)

except AuthenticationError:
    print("Authentication failed (401): Check your Gemini API key.")

except RateLimitError:
    print("Rate limit/quota error (429): Please try again later.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

# 2. Compare token usage for different text sizes

texts = [
    "What is a contract?",
    "What are the conditions for terminating a contract?",
    """
    The agreement may be terminated by either party if the other party
    commits a material breach of its obligations and fails to remedy
    such breach within thirty days after receiving written notice.
    """
]

print("\n--- Comparing Text Sizes ---")

for text in texts:

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ],
        )

        if response.usage:
            print("\nText:", text.strip())
            print("Input tokens:", response.usage.prompt_tokens)

    except Exception as e:
        print(f"Error: {e}")