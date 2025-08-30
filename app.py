import streamlit as st
import os
import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import nest_asyncio

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete
from utils import generate_graph, embedding_func

# Allow nested async loops in Streamlit
nest_asyncio.apply()

# --- Configuration ---
load_dotenv()
WORKING_DIR = "./rag-working-dir"
UPLOAD_DIR = "./uploads"


# --- RAG Initialization and Loading ---
async def initialize_rag():
    """Initializes a new LightRAG instance for building the database."""
    if "rag" not in st.session_state or st.session_state.rag is None:
        st.session_state.rag = LightRAG(
            working_dir=WORKING_DIR,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=embedding_func,
            ),
            llm_model_func=gpt_4o_mini_complete,
        )
        await st.session_state.rag.initialize_storages()
        await initialize_pipeline_status()
    return st.session_state.rag



# --- Sync wrappers for Streamlit ---
def initialize_rag_sync():
     return asyncio.run(initialize_rag())



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


# --- Streamlit UI ---
st.set_page_config(page_title="LightRAG Query App", layout="wide")
st.title("📄 LightRAG Document Query System")



# Sidebar for navigation
page = st.sidebar.radio("Choose a page", ["Build RAG Database", "Query RAG Database", "View Knowledge Graph"])




if page == "Build RAG Database":
    st.header("Step 1: Build Your RAG Database")

    uploaded_files = st.file_uploader(
        "Upload .txt files", type=["txt"], accept_multiple_files=True
    )

    if st.button("Build Database"):
        if uploaded_files:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.info(f"Saved {len(uploaded_files)} files to `{UPLOAD_DIR}`.")

            with st.spinner("Initializing RAG and building database... This may take a moment."):
                try:
                    rag = initialize_rag_sync()

                    doc_count = 0
                    processed_files = []
                    for filename in os.listdir(UPLOAD_DIR):
                        if filename.endswith(".txt"):
                            try:
                                filepath = os.path.join(UPLOAD_DIR, filename)
                                with open(filepath, "r", encoding="utf-8") as f:
                                    text_context = f.read()
                                asyncio.run(rag.ainsert(text_context))
                                doc_count += 1
                                processed_files.append(filepath)
                                st.write(f"Processed: {filename}")
                            except Exception as e:
                                st.error(f"Error processing file {filename}: {e}")
                    
                    for filepath in processed_files:
                        os.remove(filepath)

                    st.success(f"Successfully built the database with {doc_count} documents!")
                    st.balloons()
                except Exception as e:
                    st.error(f"An error occurred during database construction: {e}")
        else:
            st.warning("Please upload at least one `.txt` file.")




elif page == "Query RAG Database":
    rag = load_rag_sync()
    st.header("Step 2: Query Your RAG Database")
    question = st.text_area("Enter your questions here:", height=150)

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

                    loop = asyncio.get_event_loop()
                    answer = loop.run_until_complete(run_query())

                    st.session_state.answer = answer
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question.")

    if 'answer' in st.session_state and st.session_state.answer:
        st.header("Answer")
        st.write(st.session_state.answer)




elif page == "View Knowledge Graph":
    st.header("Knowledge Graph")
    graph_path = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")

    try:
        if os.path.exists(graph_path):
            generate_graph(graph_path)

            if os.path.exists("graph/knowledge_graph.html"):
                with open("graph/knowledge_graph.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=True)
            else:
                st.error("knowledge_graph.html not found. Please generate it first.")
        else:
            st.warning("No graph file found. Please build the database first.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# --- Sidebar Reset Button ---
if st.sidebar.button("🗑️ Reset RAG Database"):
    try:
        import shutil
        # Remove working directory and uploads
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        if os.path.exists("graph/knowledge_graph.html"):
            os.remove("graph/knowledge_graph.html")

        # Clear RAG instance from session
        if "rag" in st.session_state:
            st.session_state.rag = None

        st.sidebar.success("RAG database has been reset. You can build it again.")
    except Exception as e:
        st.sidebar.error(f"Error while resetting: {e}")


# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    "👨‍💻 Built by [Jasmeet Singh](https://github.com/jasmeetsingh-028)"
)