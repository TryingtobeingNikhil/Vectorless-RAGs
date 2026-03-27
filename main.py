import os
from vectorless.parser import parse_document
from vectorless.indexer import build_summaries
from vectorless.retriever import retrieve
from vectorless import storage
from vectorless.client import client

INDEX_PATH = "index.json"

SYSTEM_PROMPT = (
    "You are a question-answering assistant. "
    "Answer using only the context provided. "
    "If the context does not contain the answer, say so clearly."
)


def build_index(doc_path: str):
    print("Parsing document...")
    text = open(doc_path).read()
    tree = parse_document(text)

    print("Building summaries (this makes LLM calls)...")
    build_summaries(tree)

    print(f"Saving index to {INDEX_PATH}")
    storage.save(tree, INDEX_PATH)
    return tree


def ask(query: str) -> str:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Index not found. Run build_index() first.")

    tree = storage.load(INDEX_PATH)
    context = retrieve(query, tree)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        max_completion_tokens=500,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    build_index("document.md")
    print(ask("How does Python handle concurrency and the GIL?"))
