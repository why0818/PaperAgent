from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from .models import Chunk, PaperDocument


def build_knowledge_graph(
    documents: list[PaperDocument],
    chunks: list[Chunk],
    max_concepts: int = 42,
) -> dict:
    doc_by_id = {document.id: document for document in documents}
    concept_counts: Counter[str] = Counter()
    doc_concepts: dict[str, Counter[str]] = defaultdict(Counter)
    concept_sections: dict[str, Counter[str]] = defaultdict(Counter)

    for chunk in chunks:
        if chunk.document_id not in doc_by_id:
            continue
        for keyword in chunk.keywords[:8]:
            concept = normalize_concept(keyword)
            if not useful_concept(concept):
                continue
            concept_counts[concept] += 1
            doc_concepts[chunk.document_id][concept] += 1
            if chunk.section:
                concept_sections[concept][chunk.section] += 1

    top_concepts = {
        concept for concept, _ in concept_counts.most_common(max_concepts)
    }

    nodes = []
    for document in documents:
        concepts = [
            concept for concept, _ in doc_concepts[document.id].most_common(10)
            if concept in top_concepts
        ]
        nodes.append(
            {
                "id": document.id,
                "type": "document",
                "label": short_title(document.title),
                "title": document.title,
                "pages": document.pages,
                "chunks": document.chunks,
                "concepts": concepts,
            }
        )

    for concept in sorted(top_concepts):
        section = concept_sections[concept].most_common(1)
        nodes.append(
            {
                "id": f"concept:{concept}",
                "type": "concept",
                "label": concept,
                "title": concept,
                "weight": concept_counts[concept],
                "section": section[0][0] if section else "",
            }
        )

    edges = []
    for document_id, concepts in doc_concepts.items():
        for concept, weight in concepts.items():
            if concept not in top_concepts:
                continue
            edges.append(
                {
                    "source": document_id,
                    "target": f"concept:{concept}",
                    "type": "mentions",
                    "weight": weight,
                }
            )

    for left_id, right_id in combinations(doc_by_id.keys(), 2):
        shared = set(doc_concepts[left_id]) & set(doc_concepts[right_id]) & top_concepts
        if not shared:
            continue
        strength = sum(min(doc_concepts[left_id][item], doc_concepts[right_id][item]) for item in shared)
        edges.append(
            {
                "source": left_id,
                "target": right_id,
                "type": "related",
                "weight": strength,
                "shared_concepts": sorted(shared)[:12],
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "documents": len(documents),
            "concepts": len(top_concepts),
            "edges": len(edges),
        },
    }


def normalize_concept(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())[:64]


def useful_concept(value: str) -> bool:
    if len(value) < 3:
        return False
    if value.isdigit():
        return False
    if value in {
        "available",
        "data",
        "doi",
        "org",
        "https",
        "http",
        "wiley",
        "springer",
        "copyright",
        "rights",
        "reserved",
        "paper",
        "study",
        "article",
        "figure",
        "table",
        "results",
        "methods",
        "method",
        "supplementary",
    }:
        return False
    return True


def short_title(title: str, limit: int = 46) -> str:
    title = " ".join(title.split())
    if len(title) <= limit:
        return title
    return title[: limit - 1].rstrip() + "..."
