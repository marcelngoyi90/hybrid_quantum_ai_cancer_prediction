import sqlite3
import tempfile
import unittest
from pathlib import Path

from rag_system.build_database import build_database


class DatabaseBuildTests(unittest.TestCase):
    def test_database_contains_documents_and_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "knowledge.db"
            result = build_database(database)

            self.assertTrue(database.is_file())
            self.assertGreaterEqual(result["documents"], 20)
            self.assertGreaterEqual(result["chunks"], result["documents"])

            connection = sqlite3.connect(database)
            try:
                categories = {
                    row[0]
                    for row in connection.execute(
                        "SELECT DISTINCT category FROM documents"
                    )
                }
            finally:
                connection.close()

            self.assertIn("curated_research", categories)
            self.assertIn("project_primary", categories)
            self.assertIn("external_evidence", categories)


if __name__ == "__main__":
    unittest.main()
