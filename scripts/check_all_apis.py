import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import config
from models.groq_model import GroqModel
from models.openrouter_model import OpenRouterModel
from models.gemini_model import GeminiModel

def test_groq():
    print("Testing Groq...")
    try:
        model = GroqModel(model=config.GROQ_MODEL)
        if model.is_available():
            print("  Groq is available!")
            res = model.generate("Say hello", system="You are a helpful assistant")
            print(f"  Groq output: {res.content}")
        else:
            print("  Groq is not available.")
    except Exception as e:
        print(f"  Groq error: {e}")

def test_openrouter():
    print("Testing OpenRouter...")
    try:
        model = OpenRouterModel(model=config.OPENROUTER_MODEL)
        if model.is_available():
            print("  OpenRouter is available!")
            res = model.generate("Say hello", system="You are a helpful assistant")
            print(f"  OpenRouter output: {res.content}")
        else:
            print("  OpenRouter is not available.")
    except Exception as e:
        print(f"  OpenRouter error: {e}")

def test_gemini():
    print("Testing Gemini...")
    try:
        model = GeminiModel(model=config.GEMINI_MODEL)
        if model.is_available():
            print("  Gemini is available!")
            res = model.generate("Say hello", system="You are a helpful assistant")
            print(f"  Gemini output: {res.content}")
        else:
            print("  Gemini is not available.")
    except Exception as e:
        print(f"  Gemini error: {e}")

if __name__ == "__main__":
    test_groq()
    test_openrouter()
    test_gemini()
