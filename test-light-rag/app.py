import streamlit as st
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete

# Set the working directory for LightRAG
WORKING_DIR = "./rag-working-dir"

# Load the sentence transformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

async def embedding_func(texts: list[str]) -> np.ndarray:
    """Compute embeddings using SentenceTransformers."""
    return embedding_model.encode(texts, convert_to_numpy=True)

# Load the RAG model
# NOTE: Caching is temporarily disabled to work around a bug with asyncio.
# This may result in slower performance as the model is reloaded on each query.
# @st.cache_resource
def load_rag_sync():
    """Loads the LightRAG model."""
    async def load():
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
    return asyncio.run(load())

# Main app layout
st.set_page_config(layout="wide")
st.title("Crop Information")

# Sidebar for page selection
with st.sidebar:
    page = st.radio("Choose a page", ["Query RAG", "Knowledge Graph"])

if page == "Query RAG":
    rag = load_rag_sync()
    st.header("Query Light RAG")
    question = st.text_area("Enter your question about crops:", height=150)

    search_mode = st.selectbox(
        "Select search mode:",
        ("local", "global", "hybrid", "naive", "mix")
    )
    with st.expander("Search Mode Descriptions"):
        st.markdown("""
        - **local**: Focuses on context-dependent information.
        - **global**: Utilizes global knowledge.
        - **hybrid**: Combines local and global retrieval methods.
        - **naive**: Performs a basic search without advanced techniques.
        - **mix**: Integrates knowledge graph and vector retrieval.
        """)

    response_type = st.selectbox(
        "Select response type:",
        ('Multiple Paragraphs', 'Single Paragraph', 'Bullet Points')
    )

    if st.button("Get Answer"):
        if question:
            try:
                # Run the async query
                async def run_query():
                    param = QueryParam(mode=search_mode, response_type=response_type)
                    answer = await rag.aquery(question, param=param)
                    return answer

                with st.spinner("Finding the best answer..."):
                    answer = asyncio.run(run_query())
                    st.session_state.answer = answer
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question.")

    if 'answer' in st.session_state and st.session_state.answer:
        st.header("Answer")
        st.write(st.session_state.answer)

elif page == "Knowledge Graph":
    st.header("Knowledge Graph")
    try:
        with open("knowledge_graph.html", "r", encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)
    except FileNotFoundError:
        st.error("knowledge_graph.html not found. Please generate it first.")
