from __future__ import annotations

import json
from pathlib import Path

from rag_system.retrieval import retrieve


QUESTIONS = Path(__file__).resolve().parent / "evaluations" / "questions.json"


def main() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    passed = 0
    rows = []

    for item in questions:
        results = retrieve(item["question"], top_k=10)
        retrieved = " ".join(result["text"] for result in results).lower()
        terms = [term.lower() for term in item["must_find"]]
        success = all(term in retrieved for term in terms)
        passed += int(success)
        rows.append(
            {
                "id": item["id"],
                "passed": success,
                "top_source": results[0]["title"] if results else None,
            }
        )

    print(f"Retrieval checks passed: {passed}/{len(questions)}")
    for row in rows:
        marker = "PASS" if row["passed"] else "FAIL"
        print(f"{marker} {row['id']}: {row['top_source']}")


if __name__ == "__main__":
    main()
