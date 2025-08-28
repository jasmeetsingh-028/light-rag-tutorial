import os
import time
import numpy as np
import asyncio
import nest_asyncio
import shutil
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
nest_asyncio.apply()
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WORKING_DIR = "working_dir"
INPUT_FILE = "inputs/crop_file.txt"

# Ensure clean working directory
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)
os.makedirs(WORKING_DIR, exist_ok=True)

# Global model (load once to avoid 429s)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------------------------------------------
# Gemini LLM wrapper
# -------------------------------------------------------------------
async def llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """LLM function wrapper for LightRAG using Gemini API."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    if history_messages is None:
        history_messages = []

    # Build conversation context
    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    combined_prompt += f"user: {prompt}"

    try:
        # Call Gemini 2.5 Flash
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[combined_prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.1,
            ),
        )

        # Some responses may not have `.text`
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return ""   # ✅ always return string, even if empty

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return ""   # ✅ fallback to empty string


# -------------------------------------------------------------------
# Embedding function
# -------------------------------------------------------------------
async def embedding_func(texts: list[str]) -> np.ndarray:
    """Compute embeddings using SentenceTransformers (cached model)."""
    return embedding_model.encode(texts, convert_to_numpy=True)


# -------------------------------------------------------------------
# Initialize RAG
# -------------------------------------------------------------------
async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def main():
    rag = asyncio.run(initialize_rag())

    # Load knowledge base
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    rag.insert(text)

    # Run a sample query
    start = time.time()
    response = rag.query(
        query="How to cultivate wheat and what are some diseases present in it?",
        param=QueryParam(mode="hybrid", top_k=5, response_type="single line"),
    )
    end = time.time()

    print(response)
    print(f"⏱️ Time taken to execute query: {end - start:.4f} seconds")


if __name__ == "__main__":
    main()
