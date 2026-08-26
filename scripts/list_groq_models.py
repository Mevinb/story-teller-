from openai import OpenAI
import config, os

api_key = config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
client = OpenAI(api_key=api_key, base_url=config.GROQ_BASE_URL)

# List available models
models = client.models.list()
for m in models.data:
    print(m.id)
