# Vectorless RAGs

Vectorless, reasoning-based RAG. No embeddings, no vector database.''

The system parses a document into a section tree, summarizes each node, and uses an LLM to navigate the tree at query time to find the most relevant content.

## Setup (Local Ollama)

This project has been updated to run 100% locally on your machine using **Ollama**. No cloud APIs or API keys are required.

1. **Install Ollama**: Download from [ollama.com](https://ollama.com) or run `brew install ollama` on macOS.
2. **Start the background service**: Run `ollama serve` (or open the Ollama app).
3. **Download the model**: Run `ollama pull llama3`.
4. **Install Python dependencies**: 
```bash
pip install openai streamlit
```

## Usage (Streamlit UI)

The primary way to interact with the RAG is through the visual step-by-step Streamlit UI:

```bash
streamlit run app.py
```

## Usage (Python Script)

```python
from main import build_index, ask

build_index("document.md")   # run once to build the index
print(ask("Your question here"))
```

Or run directly:

```bash
python main.py
```

## Project structure

```text
app.py                streamlit UI entry point
main.py               cli entry point
vectorless/
    client.py         shared OpenAI client (configured for local Ollama)
    parser.py         splits document into section tree
    indexer.py        summarizes each node
    retriever.py      navigates tree to find relevant content
    storage.py        saves/loads index to disk
    node.py           PageNode data structure
```

## Notes

- The index is saved to `index.json`. Delete it to force a rebuild if you change `document.md`.
- Retrieval includes adjacent sibling sections when the matched leaf is short.
- All LLM calls use the local `llama3` model via the `http://localhost:11434/v1` endpoint.
