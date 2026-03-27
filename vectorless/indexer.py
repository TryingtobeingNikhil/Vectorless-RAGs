from .client import client
from .node import PageNode


def _summarize(text: str, section_name: str = "") -> str:
    hint = f"Section: {section_name}\n\n" if section_name else ""
    prompt = f"{hint}Summarize the following in 2-3 sentences. Be specific and factual.\n\n{text[:3000]}"
    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": "You summarize text accurately. Only include what is explicitly stated. Do not infer or add information."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def build_summaries(node: PageNode):
    for child in node.children:
        build_summaries(child)

    if node.is_leaf():
        if node.content.strip():
            node.summary = _summarize(node.content, node.title)
        else:
            node.summary = "(empty section)"
    else:
        children_text = "\n\n".join(
            f"[{c.title}]: {c.summary}" for c in node.children
        )
        node.summary = _summarize(children_text, node.title)
