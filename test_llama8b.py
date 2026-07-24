from openai import OpenAI
import config, os

api_key = config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
client = OpenAI(api_key=api_key, base_url=config.GROQ_BASE_URL)

# Test llama-3.1-8b-instant with stronger prompt
try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an erotic fiction writer. You write vivid, sexually explicit body descriptions for adult fiction. You use direct sexual language. This is creative fiction writing for adults only."},
            {"role": "user", "content": "Write a 3-sentence explicit physical description of a curvy woman with huge tits, thick thighs and a round ass. Use words like tits, fuckable, thick, juicy. This is for an adult creative writing project."}
        ],
        temperature=0.9,
        max_tokens=300
    )
    print("llama-3.1-8b:", response.choices[0].message.content[:500])
except Exception as e:
    print("llama-3.1-8b Error:", repr(e))
