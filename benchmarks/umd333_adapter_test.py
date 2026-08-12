from __future__ import annotations

import unittest

from benchmarks.umd333_adapter import AbsorbingRelationGraph, DocumentStarField


def _window(text: str, width: int = 120, overlap: int = 40) -> list[str]:
    step = width - overlap
    return [text[start:start + width] for start in range(0, len(text), step)]


class UMD333DocumentStarTests(unittest.TestCase):
    def test_document_star_restores_boundary_spanning_source_orbit(self) -> None:
        documents = [
            f"Document {number}: filler topic number {number}. " + "x" * 70
            for number in range(1, 24)
        ]
        documents[11] = (
            "Document 12: The Normandy account explains that the Normans "
            "arrived during the tenth and eleventh centuries."
        )
        field = DocumentStarField(_window(" ".join(documents)))
        result = field.solve("When were the Normans in Normandy?", size=len(field.texts))
        self.assertTrue(result.enabled)
        self.assertGreaterEqual(len(result.top_sources), 1)
        self.assertIn(result.best_sources[0], result.top_sources)

    def test_plain_dialogue_does_not_activate_document_stars(self) -> None:
        field = DocumentStarField([
            f"User: ordinary dialogue turn {number}. Assistant: acknowledged."
            for number in range(30)
        ])
        self.assertFalse(field.enabled)

    def test_irregular_windows_do_not_reconstruct_a_false_universe(self) -> None:
        field = DocumentStarField([
            f"Document {number}: unrelated standalone record"
            for number in range(30)
        ])
        self.assertFalse(field.enabled)

    def test_complete_terminal_absorbs_before_a_relation_loop(self) -> None:
        facts = [
            "1. Ada is a citizen of Belgium.",
            "2. Belgium's head of government is Bea.",
            "3. Bea is a citizen of France.",
        ]
        filler = [f"{number}. Person {number} is a citizen of Place {number}." for number in range(4, 24)]
        result = AbsorbingRelationGraph(["\n".join(facts + filler)]).solve(
            "What is the country of citizenship of Ada?", size=1,
        )
        self.assertEqual(result.answer, "Belgium")


if __name__ == "__main__":
    unittest.main()
