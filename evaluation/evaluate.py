"""Ragas-based evaluation runner for the RAG pipeline.

Runs the full retrieval + generation pipeline against the golden dataset and
computes four metrics:
  - ContextPrecision   (retrieval quality — are retrieved chunks relevant?)
  - ContextRecall      (retrieval quality — are all needed chunks retrieved?)
  - Faithfulness       (generation quality — does the answer use the context?)
  - AnswerRelevancy    (generation quality — does the answer address the query?)

Usage:
    uv run python -m evaluation.evaluate --dataset evaluation/golden_dataset.json
                                         --output evaluation/results.json
                                         --top-k 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET = Path("evaluation/golden_dataset.json")
DEFAULT_OUTPUT = Path("evaluation/results.json")
DEFAULT_TOP_K = 5

_METRIC_KEYS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


def load_dataset(dataset_path: Path) -> list[dict]:
    """Load the golden dataset and return its list of query entries.

    Args:
        dataset_path: Path to the golden dataset JSON file.

    Returns:
        The list stored under the top-level ``"queries"`` key.
    """
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    return data["queries"]


def build_report(
    metrics: dict[str, float],
    per_query: list[dict],
    dataset_path: str,
) -> dict:
    """Assemble the evaluation report dict (pure function — no side effects).

    Kept free of any RAG-pipeline or Ragas dependency so it can be unit-tested
    in isolation.

    Args:
        metrics: Mapping of the four Ragas metric names to their aggregate
            float scores. Missing keys default to ``0.0``.
        per_query: List of per-query result dicts (one per evaluated query).
        dataset_path: Path to the dataset that was evaluated, recorded for
            provenance.

    Returns:
        A JSON-serialisable report dict with keys ``run_timestamp``,
        ``dataset_path``, ``num_queries``, ``metrics`` and ``per_query``.
    """
    return {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "num_queries": len(per_query),
        "metrics": {key: float(metrics.get(key, 0.0)) for key in _METRIC_KEYS},
        "per_query": per_query,
    }


def _fetch_chunk_texts(source_chunk_ids: list[str]) -> list[str]:
    """Retrieve the raw chunk texts for the given IDs from Qdrant.

    Args:
        source_chunk_ids: Chunk IDs cited by the generator.

    Returns:
        A list of chunk text strings. When no IDs are supplied (e.g. a
        conversational router decision), returns ``[""]`` so downstream Ragas
        always receives a non-empty context list.
    """
    if not source_chunk_ids:
        return [""]

    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)
    points = client.retrieve(
        collection_name="rag_chunks",
        ids=source_chunk_ids,
        with_payload=True,
    )
    texts = [p.payload["text"] for p in points if p.payload and "text" in p.payload]
    return texts or [""]


def run_evaluation(dataset_path: Path, output_path: Path, top_k: int) -> dict:
    """Run the full RAG pipeline over the dataset and compute Ragas metrics.

    Args:
        dataset_path: Path to the golden dataset JSON.
        output_path: Path the JSON report is written to.
        top_k: Number of retrieval results to request per query.

    Returns:
        The report dict that was written to ``output_path``.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    from orchestration.app import run_query

    queries = load_dataset(dataset_path)

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    per_query: list[dict] = []

    for entry in queries:
        result = asyncio.run(run_query(entry["query"]))
        chunk_texts = _fetch_chunk_texts(result["source_chunk_ids"])

        questions.append(entry["query"])
        answers.append(result["answer"])
        contexts.append(chunk_texts)
        ground_truths.append(entry["expected_answer"])

        per_query.append(
            {
                "id": entry["id"],
                "query": entry["query"],
                "router_decision": result["router_decision"],
                "confidence_score": result["confidence_score"],
                "retry_count": result["retry_count"],
                "num_chunks_retrieved": len(result["source_chunk_ids"]),
            }
        )

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    result = evaluate(
        dataset,
        metrics=[
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
        ],
    )

    scores = dict(result)
    metrics = {
        "context_precision": scores.get("context_precision", 0.0),
        "context_recall": scores.get("context_recall", 0.0),
        "faithfulness": scores.get("faithfulness", 0.0),
        "answer_relevancy": scores.get("answer_relevancy", 0.0),
    }

    report = build_report(metrics, per_query, str(dataset_path))
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_summary(report)
    return report


def _print_summary(report: dict) -> None:
    """Print a formatted summary table of the report to stdout.

    Args:
        report: The report dict produced by ``build_report``.
    """
    print("\n" + "=" * 52)
    print(f"  RAG Evaluation — {report['num_queries']} queries")
    print(f"  Dataset: {report['dataset_path']}")
    print(f"  Run:     {report['run_timestamp']}")
    print("=" * 52)
    print(f"  {'Metric':<24}{'Score':>10}")
    print("  " + "-" * 34)
    for key, value in report["metrics"].items():
        label = key.replace("_", " ").title()
        print(f"  {label:<24}{value:>10.4f}")
    print("=" * 52 + "\n")


def main() -> None:
    """CLI entry point — parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
