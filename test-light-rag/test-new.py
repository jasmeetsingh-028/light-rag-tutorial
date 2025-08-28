import os
import asyncio
import textract
import numpy as np
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete
from sentence_transformers import SentenceTransformer


load_dotenv()
WORKING_DIR = "./rag-working-dir"
DOCS_DIR ="./inputs/crops"

##---------------add embedding model---------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

async def embedding_func(texts: list[str]) -> np.ndarray:
    """Compute embeddings using SentenceTransformers (cached model)."""
    return embedding_model.encode(texts, convert_to_numpy=True)


##---------------initialize RAG---------------

async def initialize_rag():
    rag = LightRAG(
        working_dir= WORKING_DIR,
        embedding_func = EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
        llm_model_func = gpt_4o_mini_complete,
    )

    # initialize storage
    await rag.initialize_storages()
    # initialize pipeline status
    await initialize_pipeline_status()

    return rag


##---------------fetch docs---------------

## read the .txt file


def main():

    ##initialize rag and insert docs
    rag = asyncio.run(initialize_rag())
    print(f"{'-' * 10}Successfully Initialized RAG{'-' * 10}")
    # rag.insert(fetch_doc())

    for filename in os.listdir(DOCS_DIR):

        try:
            filepath = os.path.join(DOCS_DIR, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                text_context = f.read()

            rag.insert(text_context)
        except Exception as e:
            print(f"Error processing {filename} file: {e}")
    
    print(f"{'-' * 10}Successfully Inserted all docs{'-' * 10}")


if __name__ == "__main__":
    main()