# Vectorless RAGs

Vectorless, reasoning-based RAG. No embeddings, no vector database.

The system parses a document into a section tree, summarizes each node, and uses an LLM to navigate the tree at query time to find the most relevant content.

## How it works

```
Document → Parser → Section Tree → Indexer (summaries) → Retriever (LLM navigation) → Answer
```

## Setup

```bash
pip install openai
cp .env.example .env
# add your OPENAI_API_KEY to .env
```

## Usage

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

```
main.py               entry point
vectorless/
    client.py         shared OpenAI client
    parser.py         splits document into section tree
    indexer.py        summarizes each node
    retriever.py      navigates tree to find relevant content
    storage.py        saves/loads index to disk
    node.py           PageNode data structure
```

## Notes

- The index is saved to `index.json`. Delete it to rebuild.
- Retrieval includes adjacent sibling sections when the matched leaf is short.
- All LLM calls use `gpt-4o` for generation and `gpt-4o-mini` for summarization and routing.
