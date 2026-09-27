import unittest

from rag_system.retrieval import database_stats, retrieve


class RetrievalTests(unittest.TestCase):
    def test_database_is_ready(self):
        stats = database_stats()
        self.assertTrue(stats["ready"])
        self.assertGreaterEqual(stats["documents"], 20)

    def test_camelyon_question_retrieves_camelyon_evidence(self):
        results = retrieve("What is Camelyon17 and what does one patch label mean?")
        joined = " ".join(item["title"].lower() for item in results[:4])
        self.assertIn("camelyon", joined)

    def test_seer_question_retrieves_node_evidence(self):
        results = retrieve(
            "What does N positive regional lymph node status mean in SEER?"
        )
        joined = " ".join(
            f"{item['title']} {item['heading']}".lower()
            for item in results[:5]
        )
        self.assertTrue("seer" in joined or "node" in joined)

    def test_project_metric_question_retrieves_results(self):
        results = retrieve(
            "What was the locked test ROC AUC for the hybrid quantum model?"
        )
        text = " ".join(item["text"] for item in results[:5])
        self.assertIn("ROC-AUC", text)

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            retrieve("  ")


if __name__ == "__main__":
    unittest.main()
