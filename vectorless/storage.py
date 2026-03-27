import json
from .node import PageNode


def save(node: PageNode, path: str):
    def to_dict(n: PageNode) -> dict:
        return {
            "title": n.title,
            "content": n.content,
            "summary": n.summary,
            "depth": n.depth,
            "children": [to_dict(c) for c in n.children],
        }
    with open(path, "w") as f:
        json.dump(to_dict(node), f, indent=2)


def load(path: str) -> PageNode:
    def from_dict(d: dict) -> PageNode:
        node = PageNode(
            title=d["title"],
            content=d["content"],
            summary=d["summary"],
            depth=d["depth"],
        )
        for child_dict in d["children"]:
            child = from_dict(child_dict)
            child.parent = node
            node.children.append(child)
        return node
    with open(path) as f:
        return from_dict(json.load(f))
