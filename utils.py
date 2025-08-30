import pipmaster as pm
import networkx as nx
from pyvis.network import Network
import random
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

async def embedding_func(texts: list[str]) -> np.ndarray:
    """Compute embeddings using SentenceTransformers."""
    return embedding_model.encode(texts, convert_to_numpy=True)



def generate_graph(path: str):

    G = nx.read_graphml(path)

    net = Network(height="100vh", notebook=True)

    net.from_nx(G)

    for node in net.nodes:
        node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if "description" in node:
            node["title"] = node["description"]

    # Add title to edges
    for edge in net.edges:
        if "description" in edge:
            edge["title"] = edge["description"]

    # Save and display the network
    net.save_graph("graph/knowledge_graph.html")