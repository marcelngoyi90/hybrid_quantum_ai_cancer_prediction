from __future__ import annotations

import json
import os
import re
from pathlib import Path

from groq import Groq

from rag_system.prompts import LEVEL_INSTRUCTIONS, SYSTEM_PROMPT
from rag_system.retrieval import DEFAULT_DATABASE, retrieve


DEFAULT_MODEL = "qwen/qwen3.8-27b"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_local_environment() -> None:
    """Load simple KEY=VALUE entries without ever logging their values."""
    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"GROQ_API_KEY", "GROQ_MODEL"} and value:
            os.environ.setdefault(key, value)


def client() -> Groq:
    load_local_environment()
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Keep the key in your local environment; "
            "never place it in code or Git."
        )
    return Groq(timeout=60.0, max_retries=1)


def format_evidence(evidence: list[dict]) -> str:
    blocks = []
    for index, item in enumerate(evidence, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[S{index}] {item['title']}",
                    f"Organisation: {item['organization']}",
                    f"Evidence level: {item['evidence_level']}",
                    f"Section: {item['heading']}",
                    item["text"],
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(
        r"\[(?:system(?:\s+instruction)?|instruction)\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    return consolidate_paragraph_citations(answer)


def consolidate_paragraph_citations(answer: str) -> str:
    citation_pattern = re.compile(
        r"\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]",
        flags=re.IGNORECASE,
    )
    paragraphs = re.split(r"(\n\s*\n)", answer)

    for index in range(0, len(paragraphs), 2):
        paragraph = paragraphs[index]
        citations: list[str] = []
        for group in citation_pattern.findall(paragraph):
            for value in re.findall(r"\bS\d+\b", group, flags=re.IGNORECASE):
                normalized = value.upper()
                if normalized not in citations:
                    citations.append(normalized)

        if not citations:
            continue

        paragraph = citation_pattern.sub("", paragraph)
        paragraph = re.sub(r"[ \t]+([.,;:])", r"\1", paragraph)
        paragraph = re.sub(r"[ \t]{2,}", " ", paragraph).rstrip()
        if not paragraph.strip():
            paragraphs[index] = ""
            continue
        paragraphs[index] = f"{paragraph} [{', '.join(citations)}]"

    return "".join(paragraphs).strip()


def cited_source_numbers(answer: str, source_count: int) -> set[int]:
    numbers: set[int] = set()
    for bracket in re.findall(r"\[([^\]]+)\]", answer):
        for value in re.findall(r"\bS(\d+)\b", bracket):
            number = int(value)
            if 1 <= number <= source_count:
                numbers.add(number)
    return numbers


def answer_question(
    question: str,
    *,
    level: str = "student",
    database_path: Path = DEFAULT_DATABASE,
    top_k: int = 7,
    concise_result: bool = False,
) -> dict:
    question = str(question).strip()
    if not question:
        raise ValueError("Question cannot be empty")
    if level not in LEVEL_INSTRUCTIONS:
        raise ValueError("level must be beginner, student or technical")

    api = client()
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    retrieval_query = question
    evidence = retrieve(
        retrieval_query,
        database_path=database_path,
        top_k=top_k,
    )
    user_content = "\n\n".join(
        [
            "Answer in clear English.",
            (
                "Explain only this image prediction to a nontechnical reader in 3–4 short sentences, "
                "at most 100 words. No introduction, headings, exam framing, analogy, lists, "
                "SEER discussion or clinical-pathway discussion. Say the model predicted a pattern; "
                "never state that tumour is confirmed. Present the supplied score as a model score, "
                "not diagnostic certainty or measured accuracy. Explicitly say even a high score "
                "can be wrong. Limit scope to this patch and mention expert review for confirmation. "
                "For a negative prediction, explain that it does not rule out cancer elsewhere. "
                "Use only one supporting citation group at the end."
                if concise_result else LEVEL_INSTRUCTIONS[level]
            ),
            f"Question:\n{question}",
            (
                f"Retrieved evidence:\n{format_evidence(evidence)}"
                if evidence
                else (
                    "Retrieved evidence:\nNo matching local evidence was found. "
                    "Give a clearly labelled general explanation if the "
                    "question is related to the project's wider subjects."
                )
            ),
        ]
    )
    response = api.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.15,
        max_completion_tokens=350 if concise_result else 1400,
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("The Groq API returned no answer")

    answer = clean_answer(answer)
    cited_numbers = cited_source_numbers(answer, len(evidence))

    sources = []
    for index, item in enumerate(evidence, start=1):
        if index not in cited_numbers:
            continue
        sources.append(
            {
                "citation": f"S{index}",
                "title": item["title"],
                "organization": item["organization"],
                "url": item["source_url"],
                "source_path": item["source_path"],
                "heading": item["heading"],
                "retrieval_score": item["score"],
            }
        )

    return {
        "answer": answer,
        "model": model,
        "language": "en",
        "level": level,
        "retrieval_query": retrieval_query,
        "sources": sources,
        "answer_mode": "grounded" if cited_numbers else "general",
        "evidence": [
            item
            for index, item in enumerate(evidence, start=1)
            if index in cited_numbers
        ],
        "warning": "Research education only—not medical diagnosis or treatment advice.",
    }
