from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = RAG_ROOT / "database" / "knowledge.db"
GITHUB_BASE = (
    "https://github.com/marcelngoyi90/"
    "hybrid_quantum_ai_cancer_prediction/blob/main/"
)

PROJECT_FILES = (
    "README.md",
    "reports/study_protocol.md",
    "reports/model_v2_development_selection.md",
    "reports/joint_v2_8d_validation.md",
    "reports/final_locked_feature_protocol.md",
    "configs/joint_model_v2.json",
    "joint_v2/README.md",
    "joint_v2/experiment_config.json",
    "joint_v2/results/development/three_seed_development_summary.json",
    "joint_v2/results/locked_test/FINAL_LOCKED_TEST_REPORT.md",
    "joint_v2/results/locked_test/final_locked_test_results.json",
)


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def external_urls(text: str) -> list[str]:
    values = re.findall(r"https://[^\s)>]+", text)
    return list(dict.fromkeys(value.rstrip(".,;") for value in values))


def markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    heading = "Overview"
    lines: list[str] = []

    for line in text.splitlines():
        if re.match(r"^#{1,4}\s+", line):
            if any(value.strip() for value in lines):
                sections.append((heading, lines))
            heading = re.sub(r"^#{1,4}\s+", "", line).strip()
            lines = []
        else:
            lines.append(line)

    if any(value.strip() for value in lines):
        sections.append((heading, lines))

    return [
        (section_heading, "\n".join(section_lines).strip())
        for section_heading, section_lines in sections
        if "\n".join(section_lines).strip()
    ]


def split_long_text(text: str, maximum: int = 1800) -> list[str]:
    paragraphs = [value.strip() for value in text.split("\n\n") if value.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > maximum:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def json_to_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def collect_documents() -> list[dict]:
    documents: list[dict] = []
    curated_root = RAG_ROOT / "sources" / "curated"

    for path in sorted(curated_root.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        urls = external_urls(text)
        documents.append(
            {
                "id": stable_id("curated", relative),
                "title": title_from_markdown(text, path.stem),
                "organization": "Project RAG research corpus",
                "category": "curated_research",
                "evidence_level": "curated_from_primary_sources",
                "source_path": relative,
                "source_url": f"{GITHUB_BASE}{relative}",
                "external_urls": urls,
                "text": text,
                "format": "markdown",
            }
        )

    for relative in PROJECT_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        documents.append(
            {
                "id": stable_id("project", relative),
                "title": (
                    title_from_markdown(text, path.stem)
                    if path.suffix == ".md"
                    else path.stem.replace("_", " ").title()
                ),
                "organization": "Hybrid Quantum AI project",
                "category": "project_primary",
                "evidence_level": "project_primary",
                "source_path": relative,
                "source_url": f"{GITHUB_BASE}{relative}",
                "external_urls": [],
                "text": text,
                "format": "markdown" if path.suffix == ".md" else "json",
            }
        )

    existing = PROJECT_ROOT / "knowledge_base" / "breast_node_evidence.json"
    if existing.is_file():
        payload = json.loads(existing.read_text(encoding="utf-8"))
        for entry in payload.get("documents", []):
            text = "\n\n".join(
                [
                    str(entry.get("retrieval_text", "")),
                    "Supported claims:\n- "
                    + "\n- ".join(entry.get("supported_claims", [])),
                    "Limitations:\n- "
                    + "\n- ".join(entry.get("limitations", [])),
                    "Topics: " + ", ".join(entry.get("topics", [])),
                ]
            ).strip()
            documents.append(
                {
                    "id": f"legacy_{entry['id']}",
                    "title": entry["title"],
                    "organization": entry["organization"],
                    "category": "external_evidence",
                    "evidence_level": "curated_external_source",
                    "source_path": "knowledge_base/breast_node_evidence.json",
                    "source_url": entry["url"],
                    "external_urls": [entry["url"]],
                    "text": text,
                    "format": "plain",
                }
            )

    return documents


def initialise(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS documents;
        DROP TABLE IF EXISTS metadata;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            organization TEXT NOT NULL,
            category TEXT NOT NULL,
            evidence_level TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_url TEXT NOT NULL,
            external_urls_json TEXT NOT NULL
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            heading TEXT NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(id),
            UNIQUE(document_id, chunk_index)
        );
        """
    )


def build_database(database_path: Path) -> dict:
    documents = collect_documents()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(database_path)) as connection:
        initialise(connection)
        chunk_count = 0

        for document in documents:
            connection.execute(
                """
                INSERT INTO documents (
                    id, title, organization, category, evidence_level,
                    source_path, source_url, external_urls_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document["id"],
                    document["title"],
                    document["organization"],
                    document["category"],
                    document["evidence_level"],
                    document["source_path"],
                    document["source_url"],
                    json.dumps(document["external_urls"], ensure_ascii=False),
                ),
            )

            if document["format"] == "markdown":
                sections = markdown_sections(document["text"])
            else:
                sections = [(document["title"], document["text"])]

            index = 0
            for heading, section_text in sections:
                for chunk in split_long_text(section_text):
                    searchable = f"{document['title']}\n{heading}\n{chunk}".strip()
                    connection.execute(
                        """
                        INSERT INTO chunks (
                            document_id, chunk_index, heading, text
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (document["id"], index, heading, searchable),
                    )
                    index += 1
                    chunk_count += 1

        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            (
                ("schema_version", "1"),
                ("document_count", str(len(documents))),
                ("chunk_count", str(chunk_count)),
            ),
        )
        connection.commit()

    return {
        "database": str(database_path),
        "documents": len(documents),
        "chunks": chunk_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local RAG database")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    arguments = parser.parse_args()
    result = build_database(arguments.database.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
