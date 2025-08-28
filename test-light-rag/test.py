import os
import time
import asyncio
import nest_asyncio
import numpy as np
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete
from openai import OpenAI

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
nest_asyncio.apply()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WORKING_DIR = "working_dir"
DOCS_DIR = "inputs/crops"   # 👈 put all your crop .txt files here

if not os.path.exists(DOCS_DIR):
    raise FileNotFoundError(f"Docs folder not found: {DOCS_DIR}")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------
# LLM wrapper
# -------------------------------------------------------------------
async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    """Use OpenAI gpt-4o-mini as the reasoning model."""
    history_messages = history_messages or []
    try:
        text = await gpt_4o_mini_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=OPENAI_API_KEY
        )
        return (text or "").strip()
    except Exception as e:
        print(f"[OpenAI llm error] {e}")
        return ""


# -------------------------------------------------------------------
# Embedding function (batched)
# -------------------------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Synchronous batch call to OpenAI embeddings."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

async def embedding_func(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed texts in batches to avoid too many retries."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            vecs = embed_batch(batch)
            all_embeddings.extend(vecs)
        except Exception as e:
            print(f"[OpenAI embed error] {e}")
            # fallback: fill with zeros if something breaks
            all_embeddings.extend([[0.0] * EMBED_DIM for _ in batch])
    return np.array(all_embeddings, dtype=np.float32)


# -------------------------------------------------------------------
# Initialize RAG
# -------------------------------------------------------------------
async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    rag = asyncio.run(initialize_rag())

    # Insert all docs from the folder
    for fname in os.listdir(DOCS_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(DOCS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"📄 Inserting: {fname}")
            rag.insert(text)

    # Run a sample query
    start = time.time()
    response = rag.query(
        query="I don’t have enough water to irrigate frequently. Between potato, rice, and soybean, which one should I choose?",
        param=QueryParam(mode="global", top_k=5, response_type="single line"),
    )
    print(response)
    print(f"⏱️ Time taken: {time.time() - start:.4f}s")


if __name__ == "__main__":
    main()
