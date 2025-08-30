# Light-RAG Streamlit Application

This project provides an interactive web application built with Streamlit to demonstrate the capabilities of the `light-rag` library. It allows users to upload text documents, build a knowledge base, and ask questions using a Retrieval-Augmented Generation (RAG) system. The application also includes a feature to visualize the underlying knowledge graph.

## Application Snapshots

Here are some snapshots of the Streamlit application:

<table>
  <tr>
    <td><img src="application-snapshots/RAG-APP build database .png" width="400"/></td>
    <td><img src="application-snapshots/RAG-APP query RAG database.png" width="400"/></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><img src="application-snapshots/RAG-APP view knowledge graph.png" width="400"/></td>
  </tr>
</table>

## Features

-   **Interactive UI**: A user-friendly web interface built with Streamlit.
-   **File Upload**: Upload one or more `.txt` files to build the knowledge base.
-   **Dynamic Knowledge Base**: Build the RAG database on the fly from the uploaded documents.
-   **Natural Language Queries**: Ask questions in natural language and get answers from the RAG system.
-   **Customizable Search**: Choose from different search modes (`local`, `global`, `hybrid`, `naive`, `mix`) and response types (`Multiple Paragraphs`, `Single Paragraph`, `Bullet Points`).
-   **Knowledge Graph Visualization**: View an interactive visualization of the knowledge graph extracted from the documents.
-   **Reset Functionality**: Easily reset the RAG database to start over.

## Project Structure

```
.
├── .python-version
├── requirements.txt
├── app.py                 # Main script to run the interactive web application
├── utils.py               # Utility functions for the application
├── uploads/               # Directory where uploaded files are temporarily stored
├── rag-working-dir/       # Stores the generated knowledge graph, vector DBs, and caches
└── graph/                 # Stores the generated knowledge graph visualization
```

## Setup

1.  **Python Version**: This project uses Python `3.12`. Ensure you have it installed and activated in your environment.

2.  **Install Dependencies**: Install the required Python packages using `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

3.  **API Keys**: The project uses OpenAI's `gpt-4o-mini` model. You need to have an OpenAI API key. Create a file named `.env` in the root directory and add your key like this:
    ```
    OPENAI_API_KEY="your-api-key-here"
    ```

## Usage

To run the web application, use the following command:

```bash
streamlit run app.py
```

This will open a new tab in your web browser with the application.

### How It Works

The application is divided into three main pages:

1.  **Build RAG Database**:
    -   Upload one or more `.txt` files.
    -   Click the "Build Database" button to process the files and create the knowledge base.

2.  **Query RAG Database**:
    -   Enter a question in the text area.
    -   Select a search mode and response type.
    -   Click "Get Answer" to see the response from the RAG system.

3.  **View Knowledge Graph**:
    -   This page displays an interactive visualization of the knowledge graph generated from the documents.

You can reset the database at any time by clicking the "Reset RAG Database" button in the sidebar.