import os
from openai import OpenAI
import config

print("Testing Groq API...")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = config.GROQ_API_KEY
    
print(f"Key begins with: {api_key[:5]}")

try:
    client = OpenAI(api_key=api_key, base_url=config.GROQ_BASE_URL)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    print("Error:", repr(e))
