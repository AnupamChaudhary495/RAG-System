"""Unit tests for Phase 6 — evaluation harness.

Golden-dataset tests are pure file/schema checks. Report-schema tests exercise
``build_report`` directly (a pure function) so neither the RAG pipeline nor
Ragas is invoked.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.evaluate import build_report

DATASET_PATH = Path("evaluation/golden_dataset.json")

VALID_QUERY_TYPES = {
    "factual",
    "comparison",
    "temporal",
    "multi_hop",
    "out_of_scope",
}


def _load_dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


class TestGoldenDatasetLoading:
    def test_dataset_file_exists(self):
        assert DATASET_PATH.exists()

    def test_dataset_has_queries_key(self):
        data = _load_dataset()
        assert "queries" in data

    def test_queries_is_nonempty_list(self):
        data = _load_dataset()
        assert isinstance(data["queries"], list)
        assert len(data["queries"]) > 0

    def test_every_entry_has_required_keys(self):
        entries = _load_dataset()["queries"]
        required = {"id", "query", "expected_answer", "query_type", "notes"}
        for entry in entries:
            assert required <= set(entry.keys()), f"missing keys in {entry}"

    def test_query_type_values_are_valid(self):
        entries = _load_dataset()["queries"]
        for entry in entries:
            assert entry["query_type"] in VALID_QUERY_TYPES

    def test_ids_are_unique(self):
        entries = _load_dataset()["queries"]
        ids = [entry["id"] for entry in entries]
        assert len(set(ids)) == len(ids)


def _sample_per_query() -> list[dict]:
    return [
        {
            "id": "q001",
            "query": "How many vacation days?",
            "router_decision": "retrieve",
            "confidence_score": 0.87,
            "retry_count": 0,
            "num_chunks_retrieved": 3,
        },
        {
            "id": "q002",
            "query": "What is the meal cap?",
            "router_decision": "retrieve",
            "confidence_score": 0.42,
            "retry_count": 1,
            "num_chunks_retrieved": 2,
        },
    ]


def _sample_metrics() -> dict[str, float]:
    return {
        "context_precision": 0.91,
        "context_recall": 0.83,
        "faithfulness": 0.78,
        "answer_relevancy": 0.88,
    }


class TestEvaluationReportSchema:
    def test_build_report_structure(self):
        report = build_report(
            _sample_metrics(), _sample_per_query(), "evaluation/golden_dataset.json"
        )
        expected_keys = {
            "run_timestamp",
            "dataset_path",
            "num_queries",
            "metrics",
            "per_query",
        }
        assert expected_keys <= set(report.keys())
        assert report["num_queries"] == 2
        assert report["dataset_path"] == "evaluation/golden_dataset.json"

    def test_metrics_keys_present(self):
        report = build_report(_sample_metrics(), _sample_per_query(), "x.json")
        metrics = report["metrics"]
        for key in (
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ):
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_metrics_default_to_zero_when_absent(self):
        report = build_report({}, _sample_per_query(), "x.json")
        assert report["metrics"]["context_precision"] == 0.0
        assert report["metrics"]["answer_relevancy"] == 0.0

    def test_per_query_entry_schema(self):
        report = build_report(_sample_metrics(), _sample_per_query(), "x.json")
        first = report["per_query"][0]
        required = {
            "id",
            "query",
            "router_decision",
            "confidence_score",
            "retry_count",
            "num_chunks_retrieved",
        }
        assert required <= set(first.keys())
