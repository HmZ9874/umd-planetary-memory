from __future__ import annotations

import unittest

from benchmarks.umd327_adapter import (
    StructuredCensusLedger,
    ledger_census_budget,
    ledger_census_orbit,
    solve_compact_numeric,
)


class UMD327AdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {"operation": "add", "operation_details": {"category": "food_expenses"}},
            {"operation": "delete", "operation_details": {"category": "food_expenses"}},
            {"operation": "add", "operation_details": {"subcategory": "artists"}},
            {"operation": None, "session_type": "no_memory"},
        ]

    def test_ledger_uses_active_structured_categories(self) -> None:
        ledger = StructuredCensusLedger(self.records)
        self.assertEqual(ledger.select("What is my total food spending?"), [0])
        self.assertEqual(ledger.select("Suggest some music by artists I like."), [2])
        self.assertEqual(ledger.select("Suggest some music by artists I like.", prefix=2), [])

    def test_ledger_budget_is_data_derived(self) -> None:
        self.assertEqual(ledger_census_budget("Total food spending", [1, 2, 3], 100), 51)
        self.assertEqual(ledger_census_orbit("Total food spending", [3, 1], [2, 3]), [3, 1, 2])

    def test_numeric_solver(self) -> None:
        payload = {
            "numeric": {"food_expenses": {"item.amount": {"sum": 100.0, "count": 8}}},
            "numeric_by_dimension": {"food_expenses": {"item.amount": {"item.expense_type": {"coffee": {"sum": 25.0, "count": 3}}}}},
            "numeric_by_week": {}, "numeric_by_period": {},
        }
        self.assertEqual(solve_compact_numeric("How much did I spend on coffee?", payload)["value"], 25.0)

    def test_mutable_state_conserves_live_origin_and_removes_deleted_items(self) -> None:
        records = [
            {"operation": "add", "date": "2025-01-01", "operation_details": {
                "category": "todo_list", "item": {"description": "alpha", "created_at": "2025-01-01"}}},
            {"operation": "add", "date": "2025-01-02", "operation_details": {
                "category": "todo_list", "item": {"description": "beta", "created_at": "2025-01-02"}}},
            {"operation": "delete", "date": "2025-01-03", "operation_details": {
                "category": "todo_list", "item": {"description": "alpha", "created_at": "2025-01-01"}}},
        ]
        ledger = StructuredCensusLedger(records)
        self.assertEqual(ledger.select("What tasks remain on my todo list?"), [1])
        self.assertEqual(ledger.select("What tasks remain on my todo list?", prefix=2), [0, 1])

    def test_query_intent_does_not_couple_all_list_categories(self) -> None:
        records = [
            {"operation": "add", "operation_details": {
                "category": "todo_list", "item": {"description": "ship patch"}}},
            {"operation": "add", "operation_details": {
                "subcategory": "already_read_list", "item": "Dune"}},
        ]
        ledger = StructuredCensusLedger(records)
        self.assertEqual(ledger.select("What remains on my todo list?"), [0])

    def test_expense_subtype_is_an_atomic_filter(self) -> None:
        records = [
            {"operation": "add", "operation_details": {
                "category": "food_expenses", "item": {"expense_type": "coffee", "amount": 4}}},
            {"operation": "add", "operation_details": {
                "category": "food_expenses", "item": {"expense_type": "lunch", "amount": 12}}},
        ]
        ledger = StructuredCensusLedger(records)
        self.assertEqual(ledger.select("How much did I spend on coffee?"), [0])

    def test_ambiguous_genre_is_bound_to_memory_utterance_domain(self) -> None:
        records = [
            {"operation": "add", "operation_details": {
                "subcategory": "genres", "item": "epic"}, "conversation": [
                    {"share_memory": True, "message": "I like epic movies."}]},
            {"operation": "add", "operation_details": {
                "subcategory": "genres", "item": "classical"}, "conversation": [
                    {"share_memory": True, "message": "I enjoy classical music."}]},
        ]
        ledger = StructuredCensusLedger(records)
        self.assertEqual(ledger.select("Can you suggest a movie?"), [0])
        self.assertEqual(ledger.select("Can you suggest some music?"), [1])

    def test_domain_is_inferred_when_utterance_omits_movie_noun(self) -> None:
        records = [
            {"operation": "add", "operation_details": {
                "subcategory": "genres", "item": "historical"}, "conversation": [
                    {"share_memory": True, "message": "I dislike historical dramas."}]},
            {"operation": "add", "operation_details": {
                "subcategory": "genres", "item": "bebop"}, "conversation": [
                    {"share_memory": True, "message": "I enjoy bebop."}]},
        ]
        ledger = StructuredCensusLedger(records)
        self.assertEqual(ledger.select("Recommend music in genres I enjoy"), [1])

    def test_music_item_hint_rejects_cross_domain_genre_noise(self) -> None:
        records = [
            {"operation": "add", "operation_details": {
                "subcategory": "genres", "item": "symphony"}},
            {"operation": "add", "operation_details": {
                "subcategory": "genres", "item": "space opera"}},
        ]
        ledger = StructuredCensusLedger(records)
        self.assertEqual(ledger.select("Suggest a movie"), [1])
        self.assertEqual(ledger.select("Suggest music"), [0])

    def test_same_day_recurrence_conserves_first_provenance(self) -> None:
        records = [
            {"operation": "add", "date": "2025-01-03", "operation_details": {
                "subcategory": "authors", "item": "Octavia Butler"}},
            {"operation": "delete", "date": "2025-01-03", "operation_details": {
                "subcategory": "authors", "item": "Octavia Butler"}},
            {"operation": "add", "date": "2025-01-03", "operation_details": {
                "subcategory": "authors", "item": "Octavia Butler"}},
        ]
        self.assertEqual(
            StructuredCensusLedger(records).select("Recommend a book by authors I like"),
            [0],
        )

    def test_calendar_update_keeps_creation_date_as_relative_anchor(self) -> None:
        records = [
            {"operation": "add", "date": "2025-01-01", "operation_details": {
                "category": "calendar_event", "item": {
                    "event_name": "Review", "date": "+10 days", "created_at": "2025-01-01"}}},
            {"operation": "update", "date": "2025-01-09", "operation_details": {
                "category": "calendar_event", "item": {
                    "event_name": "Review", "date": "+5 days", "created_at": "2025-01-01",
                    "updated_at": "2025-01-09"}}},
        ]
        ledger = StructuredCensusLedger(records)
        self.assertEqual(ledger.select("What upcoming events are on my calendar?", as_of="2025-01-04"), [0])
        self.assertEqual(ledger.select("What upcoming events are on my calendar?", as_of="2025-01-07"), [])

    def test_relative_month_horizon_is_half_open(self) -> None:
        records = [
            {"operation": "add", "date": "2025-01-01", "operation_details": {
                "category": "step_tracker", "item": {"step_count": 10}}},
            {"operation": "add", "date": "2025-03-31", "operation_details": {
                "category": "step_tracker", "item": {"step_count": 20}}},
            {"operation": "add", "date": "2025-04-01", "operation_details": {
                "category": "step_tracker", "item": {"step_count": 30}}},
        ]
        ledger = StructuredCensusLedger(records)
        self.assertEqual(
            ledger.select("Which month in the last 3 months had the most steps?"),
            [0, 1],
        )


if __name__ == "__main__":
    unittest.main()
