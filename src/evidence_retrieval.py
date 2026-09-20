from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import (
    cosine_similarity,
)


def load_knowledge_base(
    knowledge_base_path: Path,
) -> list[dict[str, Any]]:
    knowledge_base_path = Path(
        knowledge_base_path
    )

    if not knowledge_base_path.is_file():
        raise FileNotFoundError(
            "Knowledge base not found: "
            f"{knowledge_base_path}"
        )

    with knowledge_base_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise TypeError(
            "Knowledge base must contain a JSON object"
        )

    documents = payload.get("documents")

    if not isinstance(documents, list):
        raise TypeError(
            "Knowledge base documents must be a list"
        )

    if not documents:
        raise ValueError(
            "Knowledge base contains no documents"
        )

    required_fields = {
        "id",
        "title",
        "organization",
        "url",
        "retrieval_text",
        "supported_claims",
        "limitations",
        "topics",
    }

    document_ids = set()

    for document in documents:
        if not isinstance(document, dict):
            raise TypeError(
                "Every knowledge-base document "
                "must be a JSON object"
            )

        missing_fields = (
            required_fields - set(document)
        )

        if missing_fields:
            raise ValueError(
                "Knowledge-base document is missing: "
                f"{sorted(missing_fields)}"
            )

        document_id = str(document["id"])

        if document_id in document_ids:
            raise ValueError(
                "Duplicate knowledge-base document ID: "
                f"{document_id}"
            )

        document_ids.add(document_id)

        if not str(document["url"]).startswith(
            "https://"
        ):
            raise ValueError(
                "Knowledge-base URLs must use HTTPS"
            )

    return documents


def build_search_text(
    document: dict[str, Any],
) -> str:
    parts = [
        str(document["title"]),
        str(document["organization"]),
        str(document["retrieval_text"]),
    ]

    parts.extend(
        str(value)
        for value in document["topics"]
    )
    parts.extend(
        str(value)
        for value in document[
            "supported_claims"
        ]
    )

    return " ".join(parts)


def retrieve_evidence(
    query: str,
    knowledge_base_path: Path,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    query = str(query).strip()

    if not query:
        raise ValueError(
            "Retrieval query cannot be empty"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero"
        )

    documents = load_knowledge_base(
        knowledge_base_path
    )

    document_texts = [
        build_search_text(document)
        for document in documents
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    document_matrix = (
        vectorizer.fit_transform(document_texts)
    )
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        document_matrix,
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices:
        score = float(similarities[index])

        if score <= 0:
            continue

        document = documents[int(index)]

        results.append(
            {
                "rank": len(results) + 1,
                "retrieval_score": round(
                    score,
                    6,
                ),
                "id": document["id"],
                "title": document["title"],
                "organization": (
                    document["organization"]
                ),
                "url": document["url"],
                "evidence_summary": (
                    document["retrieval_text"]
                ),
                "supported_claims": (
                    document["supported_claims"]
                ),
                "limitations": (
                    document["limitations"]
                ),
            }
        )

        if len(results) == min(
            top_k,
            len(documents),
        ):
            break

    if not results:
        raise RuntimeError(
            "No relevant evidence was retrieved"
        )

    return results