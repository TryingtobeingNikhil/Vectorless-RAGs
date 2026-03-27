from .client import client
from .node import PageNode

# Leaf nodes with fewer than this many words pull in adjacent siblings for context.
_SHORT_LEAF_THRESHOLD = 80


def _pick_child(query: str, node: PageNode) -> PageNode:
    options = "\n".join(
        f"{i + 1}. [{c.title}]: {c.summary}"
        for i, c in enumerate(node.children)
    )
    prompt = f"""You are navigating a document tree to find the answer to a question.

Current section: "{node.title}"
Question: {query}

Children of this section:
{options}

Which child section most likely contains the answer? Reply with only the number."""

    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": "You navigate document trees. Reply with only the number of the best matching section."
            },
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=5,
    )
    try:
        index = int(response.choices[0].message.content.strip()) - 1
        return node.children[index]
    except (ValueError, IndexError):
        return node.children[0]


def _sibling_context(node: PageNode) -> str:
    """Return text from the previous and next siblings, if they exist and have content."""
    parent = node.parent
    if parent is None:
        return ""

    siblings = parent.children
    try:
        idx = siblings.index(node)
    except ValueError:
        return ""

    parts = []
    if idx > 0 and siblings[idx - 1].content.strip():
        parts.append(f"[{siblings[idx - 1].title}]\n{siblings[idx - 1].content.strip()}")
    if idx < len(siblings) - 1 and siblings[idx + 1].content.strip():
        parts.append(f"[{siblings[idx + 1].title}]\n{siblings[idx + 1].content.strip()}")

    return "\n\n".join(parts)


def retrieve(query: str, root: PageNode) -> str:
    node = root
    while not node.is_leaf():
        if not node.children:
            break
        node = _pick_child(query, node)

    content = node.content

    # If the leaf is short, pull in adjacent siblings for additional context.
    if len(content.split()) < _SHORT_LEAF_THRESHOLD:
        extra = _sibling_context(node)
        if extra:
            content = f"[{node.title}]\n{content.strip()}\n\n{extra}"

    return content
