"""
Vectorless RAG Demo
-------------------
A Streamlit app that demonstrates document navigation as retrieval 
traversing a structured document tree step by step instead of searching embeddings.
"""

import os
import sys
import time
import streamlit as st

from vectorless.retriever import _pick_child, _sibling_context, _SHORT_LEAF_THRESHOLD
from vectorless.node import PageNode
from vectorless.parser import parse_document
from vectorless.indexer import build_summaries
from vectorless.storage import load, save
from vectorless.client import client

INDEX_PATH = "index.json"
DOC_PATH = "document.md"



# Document tree and index logic


@st.cache_resource
def get_document_tree() -> PageNode:
    """Load or lazily build the Vectorless RAG document index."""
    if not os.path.exists(INDEX_PATH):
        if not os.path.exists(DOC_PATH):
            st.error(f"Cannot find default document at {DOC_PATH}")
            st.stop()
        
        with st.spinner("Building index for the first time... This makes LLM calls and might take a minute."):
            with open(DOC_PATH, "r", encoding="utf-8") as f:
                text = f.read()
            tree = parse_document(text)
            build_summaries(tree)
            save(tree, INDEX_PATH)
        return tree
    return load(INDEX_PATH)

# Navigation logic


def retrieve_leaf_content(node: PageNode) -> str:
    content = node.content
    if len(content.split()) < _SHORT_LEAF_THRESHOLD:
        extra = _sibling_context(node)
        if extra:
            content = f"[{node.title}]\n{content.strip()}\n\n{extra}"
    return content


def generate_answer(query: str, context: str) -> str:
    SYSTEM_PROMPT = (
        "You are a question-answering assistant. "
        "Answer using only the context provided. "
        "If the context does not contain the answer, say so clearly."
    )
    response = client.chat.completions.create(
        model="llama3",
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


def navigate(query: str, root: PageNode) -> list[dict]:
    """
    Walk the document tree top-down via LLM scoring at each step.
    Returns a list of step records for the UI to render sequentially.
    """
    steps = []
    path = []

    steps.append({"message": "Starting at document root...", "path": []})

    node = root
    while not node.is_leaf():
        if not node.children:
            break

        steps.append({
            "message": f"Scanning {len(node.children)} section{'s' if len(node.children) != 1 else ''}...",
            "path": list(path),
        })

        chosen = _pick_child(query, node)
        path.append(chosen.title)

        steps.append({
            "message": f"Selecting → {chosen.title}",
            "path": list(path),
            "selected": chosen.title,
        })

        node = chosen

    context = retrieve_leaf_content(node)

    steps.append({
        "message": f"Reached leaf node: {node.title}",
        "path": list(path),
        "leaf": node.title,
        "content": context,
    })

    return steps


# Rendering helpers


def render_tree(node: PageNode, path: list[str], depth: int = 0) -> str:
    """
    Recursively build a plain-text tree with the active path highlighted.
    Returns a single string for display inside st.code().
    """
    lines = []
    for i, child in enumerate(node.children):
        is_last = i == len(node.children) - 1
        connector = "└── " if is_last else "├── "
        indent = "    " * depth

        in_path = len(path) > depth and path[depth] == child.title
        marker = " ◀" if in_path else ""
        lines.append(f"{indent}{connector}{child.title}{marker}")

        if child.children:
            child_lines = render_tree(child, path if in_path else [], depth + 1)
            lines.append(child_lines)

    return "\n".join(lines)


def render_step_message(step: dict) -> str:
    """Format a single navigation step for display."""
    msg = step["message"]
    if step.get("selected"):
        return f"  → {msg}"
    if step.get("leaf"):
        return f"  ✓ {msg}"
    return f"  {msg}"

# ---------------------------------------------------------------------------
# Page config and styles
# ---------------------------------------------------------------------------

def configure_page():
    st.set_page_config(
        page_title="Vectorless RAG",
        page_icon="🧭",
        layout="centered",
    )
    st.markdown("""
        <style>
            /* Tighten default Streamlit padding */
            .block-container { padding-top: 2rem; padding-bottom: 2rem; }

            /* Subtle section dividers */
            hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0; }

            /* Step trace styling */
            .step-line {
                font-family: 'Courier New', monospace;
                font-size: 0.88rem;
                color: #333;
                padding: 2px 0;
            }
            .step-leaf {
                color: #1a6e3c;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    configure_page()

    st.title("🧭 Vectorless RAG Demo")
    st.caption("Navigating document structure instead of searching embeddings.")

    st.markdown("---")

    st.markdown("""
        **Suggested questions to test the demo:**
        - *What computer alarms went off during the Eagle's descent?*
        - *Who remained alone in the Command Module Columbia?*
        - *How many pounds of thrust did the Saturn V rocket generate?*
    """)
    
    query = st.text_input(
        "Your question",
        placeholder="e.g. How does the model process data?",
        label_visibility="visible",
    )

    run = st.button("Run", type="primary", disabled=not query.strip())

    try:
        document_tree = get_document_tree()
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        return

    if not run:
        st.subheader("Document Tree")
        st.code("root\n" + render_tree(document_tree, []), language=None)
        return

    query = query.strip()
    steps = navigate(query, document_tree)

    st.markdown("---")

    st.subheader("Navigation Trace")
    trace_container = st.empty()
    rendered_lines: list[str] = []
    current_path: list[str] = []
    leaf: str | None = None
    leaf_content: str | None = None

    st.subheader("Document Tree")
    tree_container = st.empty()

    tree_container.code(
        "root\n" + render_tree(document_tree, []),
        language=None,
    )

    st.markdown("---")

    # Stream steps with delay
    for step in steps:
        line = render_step_message(step)
        rendered_lines.append(line)
        current_path = step.get("path", current_path)

        if step.get("leaf"):
            leaf = step["leaf"]
            leaf_content = step.get("content")

        # Update trace block
        trace_container.code("\n".join(rendered_lines), language=None)

        # Update tree highlight
        tree_container.code(
            "root\n" + render_tree(document_tree, current_path),
            language=None,
        )

        time.sleep(0.7)

    st.subheader("Retrieved Context")
    if leaf_content:
        st.info(leaf_content)
    else:
        st.info("No context retrieved.")

    st.subheader("Generated Answer")
    if leaf_content:
        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(query, leaf_content)
                st.success(answer)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
    else:
        st.success("Answer unavailable for this section.")

    st.markdown("---")
    st.caption(f"Path taken: {' → '.join(current_path)}")


if __name__ == "__main__":
    main()
