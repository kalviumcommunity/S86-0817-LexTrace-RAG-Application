import os

from dotenv import load_dotenv
from openai import OpenAI

from prompts.answer import render_prompt

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
    api_key=api_key
)


# Dummy context for testing purpose
context = """
The agreement may be terminated by either party if the other party
commits a material breach and fails to remedy the breach within
30 days of receiving written notice.
"""

question = "When can the agreement be terminated?"


# Fill the template with dynamic values
prompt = render_prompt(
    context=context,
    question=question
)

print("\n--- Rendered Prompt ---")
print(prompt)

# Send the completed prompt to Gemini
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0
)

answer = response.choices[0].message.content

print("\n--- Gemini Response ---")
print(answer)