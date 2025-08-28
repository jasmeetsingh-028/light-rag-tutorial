import os
import asyncio
import time
import textract
import numpy as np
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete
from sentence_transformers import SentenceTransformer


WORKING_DIR = "./rag-working-dir"


##---------------add embedding model---------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

async def embedding_func(texts: list[str]) -> np.ndarray:
    """Compute embeddings using SentenceTransformers (cached model)."""
    return embedding_model.encode(texts, convert_to_numpy=True)



#--------Load RAG--------


async def load_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=512,
            func=embedding_func,
        ),
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

#--------Query RAG--------

async def query_rag(rag, question: str):

    start = time.time()

    # response = rag.query(
    #     query="How to cultivate wheat and what are some diseases present in it?",
    #     param=QueryParam(mode="hybrid", top_k=5, response_type="single line"),
    # )
    answer = await rag.aquery(
        question,
        param=QueryParam(mode="global", response_type='Single Paragraph')   # can be "local", "global", etc.
    )
    end = time.time()
    print(f"⏱️ Time taken to execute query (Global mode): {end - start:.4f}s")
    print(f"Q: {question}\nA: {answer}")



async def main():
    rag = await load_rag()

    await query_rag(rag, "I dont have enough water to irrigate frequently. Between potato, rice, and soybean, which one should I choose?")

if __name__ == "__main__":
    asyncio.run(main())