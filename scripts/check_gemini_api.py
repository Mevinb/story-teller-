import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import config
from models.gemini_model import GeminiModel

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
    test_gemini()
