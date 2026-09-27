import unittest
from unittest.mock import MagicMock, patch

from rag_system.llm import (
    answer_question,
    cited_source_numbers,
    clean_answer,
    consolidate_paragraph_citations,
)
from rag_system.prompts import SYSTEM_PROMPT


class AnswerSafetyTests(unittest.TestCase):
    def test_prompt_allows_broader_related_questions(self):
        self.assertIn("any question related to the project", SYSTEM_PROMPT)
        self.assertIn("General explanation:", SYSTEM_PROMPT)

    @patch("rag_system.llm.retrieve", return_value=[])
    @patch("rag_system.llm.client")
    def test_open_question_can_use_labelled_general_knowledge(
        self,
        client,
        _retrieve,
    ):
        response = MagicMock()
        response.choices[0].message.content = (
            "General explanation: This is a broader educational answer."
        )
        client.return_value.chat.completions.create.return_value = response
        result = answer_question("Explain a related AI concept")
        self.assertEqual(result["answer_mode"], "general")
        self.assertEqual(result["sources"], [])

    def test_hidden_instruction_labels_are_removed(self):
        value = clean_answer(
            "Do not combine the scores [System Instruction]. Keep [S1]."
        )
        self.assertNotIn("System Instruction", value)
        self.assertIn("[S1]", value)

    def test_grouped_citations_are_recognised(self):
        self.assertEqual(
            cited_source_numbers("Supported paragraph [S1, S4].", 7),
            {1, 4},
        )

    def test_citations_are_consolidated_once_per_paragraph(self):
        value = consolidate_paragraph_citations(
            "First claim [S1]. Second claim [S1, S4].\n\nNext [S2]."
        )
        self.assertEqual(value.count("[S1, S4]"), 1)
        self.assertNotIn("[S1].", value)
        self.assertEqual(value.count("[S2]"), 1)

    def test_citation_only_paragraph_is_removed(self):
        value = consolidate_paragraph_citations(
            "Supported explanation [S1].\n\n[S1] [S2]"
        )
        self.assertEqual(value, "Supported explanation. [S1]")


if __name__ == "__main__":
    unittest.main()
