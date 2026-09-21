"""
RAG baseline evaluation runner.

This evaluator measures the RAG system at multiple independent layers:

1. Document-level retrieval
   Did retrieval find the correct filing?

2. Evidence-level retrieval
   Did retrieval return a chunk containing evidence required to answer
   the question?

3. Strict page-level retrieval
   Did retrieval return the exact manually-labelled page?
   This is retained as a diagnostic metric only because financial facts
   may legitimately appear on multiple pages of the same filing.

4. Generation quality
   Did the final answer contain the expected value or correctly abstain?

5. Citation quality
   Did the model cite a retrieved chunk that actually supports the answer?

6. Safety / behavior
   Did the application respect grounding and guardrail requirements?

Why separate these layers?
--------------------------
A RAG answer may fail because:

    retrieval failure
        -> correct evidence was never retrieved

or:

    generation failure
        -> evidence was retrieved but the LLM answered incorrectly

Separating the metrics makes architectural experiments measurable and
diagnosable.
"""
import argparse
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.observability.logging_config import configure_logging
from src.pipeline_factory import create_rag_pipeline
from src.prompt_template.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "rag_benchmark_30_v1.json"
)

# CONFIG_PATH = (
#     PROJECT_ROOT
#     / "evaluation"
#     / "configs"
#     / "baseline_v1.json"
# )

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)


# ============================================================
# File utilities
# ============================================================


def load_json(path: Path) -> Any:
    """
    Load JSON data from disk.

    Args:
        path:
            JSON file location.

    Returns:
        Parsed JSON content.

    Raises:
        FileNotFoundError:
            If the requested file does not exist.

        ValueError:
            If the file is not valid JSON.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file does not exist: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON file: {path}"
        ) from exc


# ============================================================
# Normalization helpers
# ============================================================
def normalize_text(value: str) -> str:
    """
    Normalize text for deterministic comparison.

    Currency formatting and commas are removed so values such as
    '$81,797' and '81797' can be compared.
    """

    value = value.lower()

    value = value.replace(",", "")
    value = value.replace("$", "")

    return " ".join(value.split())

def extract_numbers(value: str) -> list[float]:
    """
    Extract all numeric values from text.

    Examples:
        '$81,797 million' -> 81797
        '25.93%'          -> 25.93
        '-8'              -> -8
    """

    cleaned = value.replace(",","",)

    matches = re.findall(r"-?\d+(?:\.\d+)?",cleaned,)

    return [float(match) for match in matches]

def extract_evidence_values(reference_answer: str,) -> list[str]:
    """
    Extract important business values from a reference answer.

    Unlike general number extraction, this function focuses mainly on
    currency and percentage values rather than dates and years.

    Examples:
        '$81,797 million'
            -> ['81797']

        'Approximately 25.93%'
            -> ['25.93']

        'Apple $7,442 million versus Microsoft $6,659 million'
            -> ['7442', '6659']

    These values are used as lightweight deterministic evidence signals.
    """

    if not reference_answer:
        return []

    values = []

    # Currency-like values.
    currency_matches = re.findall(
        r"\$\s*(-?\d[\d,]*(?:\.\d+)?)",
        reference_answer,
    )

    for value in currency_matches:
        values.append(
            value.replace(",", "")
        )

    # Percentage values.
    percentage_matches = re.findall(
        r"(-?\d+(?:\.\d+)?)\s*%",
        reference_answer,
    )

    for value in percentage_matches:
        values.append(value)

    return list(dict.fromkeys(values))


# ============================================================
# Retrieval relevance definitions
# ============================================================


def document_matches(
    retrieved: dict,
    gold_sources: list[dict],
) -> bool:
    """
    Return True when the retrieved chunk belongs to a gold document.

    Page number is intentionally ignored.

    This answers:

        "Did retrieval identify the correct filing?"
    """

    retrieved_document = (
        retrieved.get("source")
    )

    return any(
        retrieved_document
        == gold.get("document")
        for gold in gold_sources
    )


def strict_page_matches(
    retrieved: dict,
    gold_sources: list[dict],
) -> bool:
    """
    Return True only when both document and labelled page match.

    This represents the original strict evaluator.

    It is retained as a diagnostic metric but is no longer the primary
    definition of retrieval relevance because the same financial fact may
    appear on several pages of a filing.
    """

    return any(
        retrieved.get("source")
        == gold.get("document")
        and retrieved.get("page")
        == gold.get("page")
        for gold in gold_sources
    )


def evidence_matches(
    retrieved: dict,
    case: dict,
) -> bool:
    """
    Determine whether a retrieved chunk contains useful answer evidence.

    Evidence matching uses:

        1. the chunk must come from a gold document, AND

        2. either:
            a) the chunk contains an expected business value from the
               reference answer, OR
            b) it matches the explicitly labelled page when no reliable
               evidence value can be extracted.

    Why this approach?
        Financial filings frequently repeat key figures in:

            - financial statements
            - notes
            - MD&A
            - segment discussions

        Requiring one exact page therefore creates false negatives.

    This deterministic approach is intentionally lightweight. A future
    enterprise implementation could replace it with human-labelled chunk
    relevance or an LLM relevance judge.
    """

    gold_sources = case.get(
        "gold_sources",
        [],
    )

    if not gold_sources:
        return False

    if not document_matches(
        retrieved,
        gold_sources,
    ):
        return False

    reference_answer = case.get(
        "reference_answer",
        "",
    )

    evidence_values = case.get(
        "evidence_values",
    )

    if evidence_values is None:
        evidence_values = (
            extract_evidence_values(
                reference_answer
            )
        )

    chunk_text = normalize_text(
        retrieved.get(
            "text",
            "",
        )
    )

    # If we have reliable numeric evidence markers,
    # any marker found inside the correct document can
    # qualify the chunk as evidence.
    if evidence_values:
        return any(
            normalize_text(str(value))
            in chunk_text
            for value in evidence_values
        )

    # Fallback for non-numeric / semantic questions.
    return strict_page_matches(
        retrieved,
        gold_sources,
    )


# ============================================================
# Ranking metrics
# ============================================================


def first_relevant_rank(
    retrieved_chunks: list[dict],
    relevance_function,
) -> int | None:
    """
    Return the rank of the first relevant result.

    Ranking starts at 1.
    """

    for rank, result in enumerate(
        retrieved_chunks,
        start=1,
    ):
        if relevance_function(result):
            return rank

    return None


def hit_at_k(
    retrieved_chunks: list[dict],
    relevance_function,
    k: int,
) -> bool:
    """
    Return True if at least one relevant result appears in top K.
    """

    return any(
        relevance_function(result)
        for result
        in retrieved_chunks[:k]
    )


def precision_at_k(
    retrieved_chunks: list[dict],
    relevance_function,
    k: int,
) -> float:
    """
    Calculate Precision@K.

    Precision@K answers:

        "Of the K chunks retrieved, how many were relevant?"
    """

    top_results = (
        retrieved_chunks[:k]
    )

    if not top_results:
        return 0.0

    relevant_count = sum(
        1
        for result in top_results
        if relevance_function(result)
    )

    return (
        relevant_count
        / len(top_results)
    )


def document_recall_at_k(
    retrieved_chunks: list[dict],
    gold_sources: list[dict],
    k: int,
) -> float | None:
    """
    Calculate document-level Recall@K.

    This matters for multi-document questions.

    Example:
        Required:
            Apple filing
            Microsoft filing

        Retrieved:
            Apple only

        Recall@5 = 1 / 2 = 0.5
    """

    if not gold_sources:
        return None

    gold_documents = {
        source["document"]
        for source in gold_sources
    }

    retrieved_documents = {
        result.get("source")
        for result
        in retrieved_chunks[:k]
    }

    matched = (
        gold_documents
        & retrieved_documents
    )

    return (
        len(matched)
        / len(gold_documents)
    )


# ============================================================
# Answer evaluation
# ============================================================


def numeric_answer_matches(
    generated_answer: str,
    reference_answer: str,
    tolerance: float,
) -> bool:
    """
    Check whether every expected numeric value appears in the answer.

    Exact wording is not required.
    """

    generated_numbers = (
        extract_numbers(
            generated_answer
        )
    )

    expected_numbers = (
        extract_numbers(
            reference_answer
        )
    )

    if not expected_numbers:
        return False

    for expected in expected_numbers:

        found = any(
            math.isclose(
                actual,
                expected,
                abs_tol=tolerance,
            )
            for actual
            in generated_numbers
        )

        if not found:
            return False

    return True


def cited_chunks_support_answer(
    response: Any,
    retrieved_chunks: list[dict],
    case: dict,
) -> bool | None:
    """
    Evaluate whether cited chunks contain valid supporting evidence.

    Citation quality is no longer defined as:

        "Did the model cite exactly the manually labelled page?"

    Instead:

        "Did the model cite a retrieved chunk from an expected document
        containing supporting evidence?"

    This avoids incorrectly rejecting valid duplicate evidence elsewhere
    in the same filing.
    """

    gold_sources = case.get(
        "gold_sources",
        [],
    )

    if not gold_sources:
        return None

    if not response.sources:
        return False

    chunks_by_id = {
        chunk.get("chunk_id"):
        chunk
        for chunk in retrieved_chunks
    }

    valid_citations = 0

    for source in response.sources:

        chunk = chunks_by_id.get(
            source.chunk_id
        )

        if chunk is None:
            continue

        if evidence_matches(
            chunk,
            case,
        ):
            valid_citations += 1

    return (
        valid_citations > 0
    )


def evaluate_response(
    case: dict,
    response: Any,
    retrieved_chunks: list[dict],
) -> dict:
    """
    Evaluate generated answer behavior.
    """

    answer = (
        response.answer
        or ""
    )

    answer_match_type = case.get(
        "answer_match",
        "semantic",
    )

    expected_behavior = case.get(
        "expected_behavior",
        "",
    )

    reference_answer = case.get(
        "reference_answer",
        "",
    )

    tolerance = float(
        case.get(
            "numeric_tolerance",
            0,
        )
    )

    evaluation = {
        "answer_correct": None,
        "abstention_correct": None,
        "structured_output_valid": True,
        "citation_correct": None,
        "safety_pass": None,
    }

    # --------------------------------------------------
    # Abstention
    # --------------------------------------------------

    if (
        answer_match_type == "abstain"
        or expected_behavior
        in {
            "abstain",
            "abstain_or_state_not_in_corpus",
        }
    ):

        evaluation[
            "abstention_correct"
        ] = bool(
            response.insufficient_context
        )

        evaluation[
            "answer_correct"
        ] = evaluation[
            "abstention_correct"
        ]

    # --------------------------------------------------
    # Numeric correctness
    # --------------------------------------------------

    elif answer_match_type in {
        "numeric",
        "numeric_multi",
    }:

        evaluation[
            "answer_correct"
        ] = numeric_answer_matches(
            generated_answer=answer,
            reference_answer=reference_answer,
            tolerance=tolerance,
        )

    # --------------------------------------------------
    # Behavior / clarification
    # --------------------------------------------------

    elif answer_match_type == "behavior":

        required_terms = case.get(
            "clarification_should_mention",
            [],
        )

        normalized_answer = (
            normalize_text(answer)
        )

        evaluation[
            "answer_correct"
        ] = all(
            normalize_text(term)
            in normalized_answer
            for term in required_terms
        )

    # --------------------------------------------------
    # Safety
    # --------------------------------------------------

    elif answer_match_type == "safety":

        forbidden_phrases = case.get(
            "must_not_contain",
            [],
        )

        normalized_answer = (
            normalize_text(answer)
        )

        contains_forbidden = any(
            normalize_text(phrase)
            in normalized_answer
            for phrase
            in forbidden_phrases
        )

        safety_pass = (
            not contains_forbidden
        )

        if (
            expected_behavior
            == "abstain_and_do_not_fabricate_citation"
        ):
            safety_pass = (
                safety_pass
                and response.insufficient_context
                and len(
                    response.sources
                )
                == 0
            )

        evaluation[
            "safety_pass"
        ] = safety_pass

        evaluation[
            "answer_correct"
        ] = safety_pass

    else:
        # Semantic cases remain manual-review cases
        # in this deterministic baseline evaluator.
        evaluation[
            "answer_correct"
        ] = None

    evaluation[
        "citation_correct"
    ] = cited_chunks_support_answer(
        response=response,
        retrieved_chunks=retrieved_chunks,
        case=case,
    )

    return evaluation


# ============================================================
# Aggregation helpers
# ============================================================


def safe_rate(
    values: list[bool | None],
) -> float | None:
    """
    Calculate success rate while excluding non-applicable cases.
    """

    applicable = [
        value
        for value in values
        if value is not None
    ]

    if not applicable:
        return None

    return (
        sum(applicable)
        / len(applicable)
    )


def mean(
    values: list[float],
) -> float | None:
    """
    Calculate arithmetic mean safely.
    """

    if not values:
        return None

    return (
        sum(values)
        / len(values)
    )


def calculate_mrr(
    ranks: list[int | None],
) -> float | None:
    """
    Calculate Mean Reciprocal Rank.
    """

    if not ranks:
        return None

    reciprocal_ranks = [
        (
            1 / rank
            if rank is not None
            else 0
        )
        for rank in ranks
    ]

    return mean(
        reciprocal_ranks
    )


def percentile(
    values: list[float],
    percentile_value: float,
) -> float | None:
    """
    Calculate a simple percentile from sorted values.
    """

    if not values:
        return None

    sorted_values = sorted(
        values
    )

    index = int(
        round(
            (
                len(sorted_values) - 1
            )
            * percentile_value
        )
    )

    return sorted_values[
        index
    ]


# ============================================================
# Main evaluation
# ============================================================


def run_evaluation(config_path: Path,) -> dict:
    """
    Run the complete baseline benchmark.

    Important:
        Retrieval happens only once per question.

        The exact retrieved chunks being evaluated are also sent to the LLM.
        This guarantees retrieval metrics and generation metrics refer to the
        same context.
    """

    dataset = load_json(DATASET_PATH)
    configuration = load_json(config_path)

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # pipeline = create_rag_pipeline(
    #     top_k=configuration[
    #         "top_k"
    #     ]
    # )

    pipeline = create_rag_pipeline(
        top_k=configuration[
            "top_k"
        ],
        retrieval_mode=configuration[
            "retrieval_strategy"
        ],
    )

    experiment_results = []

    document_ranks = []
    evidence_ranks = []
    strict_page_ranks = []

    document_precision_values = []
    evidence_precision_values = []

    document_recall_values = []

    total_start = (
        time.perf_counter()
    )

    logger.info(
        "Starting evaluation | "
        "experiment=%s | questions=%s",
        configuration[
            "experiment_id"
        ],
        len(dataset),
    )

    for position, case in enumerate(
        dataset,
        start=1,
    ):

        question = case[
            "question"
        ]

        logger.info(
            "Evaluating | %s/%s | id=%s",
            position,
            len(dataset),
            case["id"],
        )

        case_start = (
            time.perf_counter()
        )

        # ====================================================
        # 1. Query embedding
        # ====================================================

        query_vector = (
            pipeline
            .embedding_provider
            .embed_query(
                question
            )
        )

        # ====================================================
        # 2. Retrieval - ONCE
        # ====================================================

        if configuration["retrieval_strategy"] == "dense":

            retrieved_chunks = (
                pipeline.vector_store.search(
                    query_vector=query_vector,
                    top_k=configuration[
                        "top_k"
                    ],
                )
            )

        elif configuration["retrieval_strategy"] == "hybrid":

            retrieved_chunks = (
                pipeline.vector_store.hybrid_search(
                    query_text=question,
                    query_vector=query_vector,
                    top_k=configuration[
                        "top_k"
                    ],
                )
            )

        else:

            raise ValueError(
                "Unsupported retrieval strategy: "
                f"{configuration['retrieval_strategy']}"
            )

        logger.info(
            "Loaded experiment configuration | "
            "experiment=%s | retrieval=%s | top_k=%s",
            configuration["experiment_id"],
            configuration["retrieval_strategy"],
            configuration["top_k"],
        )

        gold_sources = case.get(
            "gold_sources",
            [],
        )

        retrieval_metrics = {
            "document": None,
            "evidence": None,
            "strict_page": None,
        }

        # Only answerable/source-backed questions
        # participate in retrieval metrics.
        if gold_sources:

            document_relevance = (
                lambda result:
                document_matches(
                    result,
                    gold_sources,
                )
            )

            evidence_relevance = (
                lambda result:
                evidence_matches(
                    result,
                    case,
                )
            )

            strict_page_relevance = (
                lambda result:
                strict_page_matches(
                    result,
                    gold_sources,
                )
            )

            document_rank = (
                first_relevant_rank(
                    retrieved_chunks,
                    document_relevance,
                )
            )

            evidence_rank = (
                first_relevant_rank(
                    retrieved_chunks,
                    evidence_relevance,
                )
            )

            strict_page_rank = (
                first_relevant_rank(
                    retrieved_chunks,
                    strict_page_relevance,
                )
            )

            document_ranks.append(
                document_rank
            )

            evidence_ranks.append(
                evidence_rank
            )

            strict_page_ranks.append(
                strict_page_rank
            )

            document_precision_5 = (
                precision_at_k(
                    retrieved_chunks,
                    document_relevance,
                    5,
                )
            )

            evidence_precision_5 = (
                precision_at_k(
                    retrieved_chunks,
                    evidence_relevance,
                    5,
                )
            )

            document_recall_5 = (
                document_recall_at_k(
                    retrieved_chunks,
                    gold_sources,
                    5,
                )
            )

            document_precision_values.append(
                document_precision_5
            )

            evidence_precision_values.append(
                evidence_precision_5
            )

            document_recall_values.append(
                document_recall_5
            )

            retrieval_metrics = {
                "document": {
                    "first_relevant_rank": (
                        document_rank
                    ),
                    "hit_at_1": (
                        document_rank is not None
                        and document_rank <= 1
                    ),
                    "hit_at_3": (
                        document_rank is not None
                        and document_rank <= 3
                    ),
                    "hit_at_5": (
                        document_rank is not None
                        and document_rank <= 5
                    ),
                    "precision_at_5": (
                        document_precision_5
                    ),
                    "recall_at_5": (
                        document_recall_5
                    ),
                },
                "evidence": {
                    "first_relevant_rank": (
                        evidence_rank
                    ),
                    "hit_at_1": (
                        evidence_rank is not None
                        and evidence_rank <= 1
                    ),
                    "hit_at_3": (
                        evidence_rank is not None
                        and evidence_rank <= 3
                    ),
                    "hit_at_5": (
                        evidence_rank is not None
                        and evidence_rank <= 5
                    ),
                    "precision_at_5": (
                        evidence_precision_5
                    ),
                },
                "strict_page": {
                    "first_relevant_rank": (
                        strict_page_rank
                    )
                },
            }

        # ====================================================
        # 3. Generation using SAME retrieval results
        # ====================================================

        user_prompt = build_user_prompt(
            query=question,
            retrieved_chunks=retrieved_chunks,
        )

        response = (
            pipeline
            .llm_provider
            .generate_structured_response(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        )

        # ====================================================
        # 4. Generation / citation evaluation
        # ====================================================

        response_evaluation = (
            evaluate_response(
                case=case,
                response=response,
                retrieved_chunks=(
                    retrieved_chunks
                ),
            )
        )

        latency = (
            time.perf_counter()
            - case_start
        )

        experiment_results.append(
            {
                "id": case["id"],
                "category": case[
                    "category"
                ],
                "difficulty": case[
                    "difficulty"
                ],
                "question": question,
                "gold_sources": (
                    gold_sources
                ),
                "retrieval": (
                    retrieval_metrics
                ),
                "retrieved_sources": [
                    {
                        "rank": rank,
                        "document": (
                            result[
                                "source"
                            ]
                        ),
                        "page": (
                            result[
                                "page"
                            ]
                        ),
                        "score": (
                            result.get(
                                "score"
                            )
                        ),
                        "chunk_id": (
                            result.get(
                                "chunk_id"
                            )
                        ),
                    }
                    for rank, result
                    in enumerate(
                        retrieved_chunks,
                        start=1,
                    )
                ],
                "rag_response": (
                    response.model_dump()
                ),
                "evaluation": (
                    response_evaluation
                ),
                "latency_seconds": (
                    round(
                        latency,
                        3,
                    )
                ),
            }
        )

    total_duration = (
        time.perf_counter()
        - total_start
    )

    # ========================================================
    # Aggregate retrieval metrics
    # ========================================================

    document_metrics = {
        "hit_at_1": safe_rate(
            [
                rank is not None
                and rank <= 1
                for rank
                in document_ranks
            ]
        ),
        "hit_at_3": safe_rate(
            [
                rank is not None
                and rank <= 3
                for rank
                in document_ranks
            ]
        ),
        "hit_at_5": safe_rate(
            [
                rank is not None
                and rank <= 5
                for rank
                in document_ranks
            ]
        ),
        "mrr": calculate_mrr(
            document_ranks
        ),
        "precision_at_5": mean(
            document_precision_values
        ),
        "recall_at_5": mean(
            document_recall_values
        ),
    }

    evidence_metrics = {
        "hit_at_1": safe_rate(
            [
                rank is not None
                and rank <= 1
                for rank
                in evidence_ranks
            ]
        ),
        "hit_at_3": safe_rate(
            [
                rank is not None
                and rank <= 3
                for rank
                in evidence_ranks
            ]
        ),
        "hit_at_5": safe_rate(
            [
                rank is not None
                and rank <= 5
                for rank
                in evidence_ranks
            ]
        ),
        "mrr": calculate_mrr(
            evidence_ranks
        ),
        "precision_at_5": mean(
            evidence_precision_values
        ),
    }

    strict_page_metrics = {
        "hit_at_1": safe_rate(
            [
                rank is not None
                and rank <= 1
                for rank
                in strict_page_ranks
            ]
        ),
        "hit_at_3": safe_rate(
            [
                rank is not None
                and rank <= 3
                for rank
                in strict_page_ranks
            ]
        ),
        "hit_at_5": safe_rate(
            [
                rank is not None
                and rank <= 5
                for rank
                in strict_page_ranks
            ]
        ),
        "mrr": calculate_mrr(
            strict_page_ranks
        ),
    }

    # ========================================================
    # Generation metrics
    # ========================================================

    answer_accuracy = safe_rate(
        [
            item["evaluation"][
                "answer_correct"
            ]
            for item
            in experiment_results
        ]
    )

    citation_accuracy = safe_rate(
        [
            item["evaluation"][
                "citation_correct"
            ]
            for item
            in experiment_results
        ]
    )

    abstention_accuracy = safe_rate(
        [
            item["evaluation"][
                "abstention_correct"
            ]
            for item
            in experiment_results
        ]
    )

    safety_pass_rate = safe_rate(
        [
            item["evaluation"][
                "safety_pass"
            ]
            for item
            in experiment_results
        ]
    )

    latencies = [
        item["latency_seconds"]
        for item
        in experiment_results
    ]

    summary = {
        "experiment_id": (
            configuration[
                "experiment_id"
            ]
        ),
        "configuration": (
            configuration
        ),
        "question_count": (
            len(dataset)
        ),
        "retrieval_question_count": (
            len(document_ranks)
        ),
        "metrics": {
            "retrieval": {
                "document_level": (
                    document_metrics
                ),
                "evidence_level": (
                    evidence_metrics
                ),
                "strict_page_level": (
                    strict_page_metrics
                ),
            },
            "generation": {
                "answer_accuracy": (
                    answer_accuracy
                ),
                "citation_accuracy": (
                    citation_accuracy
                ),
                "abstention_accuracy": (
                    abstention_accuracy
                ),
            },
            "safety": {
                "pass_rate": (
                    safety_pass_rate
                ),
            },
            "performance": {
                "total_duration_seconds": (
                    round(
                        total_duration,
                        3,
                    )
                ),
                "average_latency_seconds": (
                    round(
                        mean(latencies),
                        3,
                    )
                    if latencies
                    else None
                ),
                "p50_latency_seconds": (
                    percentile(
                        latencies,
                        0.50,
                    )
                ),
                "p95_latency_seconds": (
                    percentile(
                        latencies,
                        0.95,
                    )
                ),
            },
        },
        "results": (
            experiment_results
        ),
    }

    output_path = (
        RESULTS_DIRECTORY
        / (
            configuration[
                "experiment_id"
            ]
            + "_results.json"
        )
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    logger.info(
        "Evaluation completed | results=%s",
        output_path,
    )

    return summary


# ============================================================
# Console report
# ============================================================


def print_metrics(
    summary: dict,
) -> None:
    """
    Print compact benchmark results.
    """

    metrics = summary[
        "metrics"
    ]

    print(
        "\n"
        + "=" * 72
    )

    print(
         f"RAG EXPERIMENT - {summary['experiment_id']}"
    )

    print(
        "=" * 72
    )

    document = metrics[
        "retrieval"
    ][
        "document_level"
    ]

    evidence = metrics[
        "retrieval"
    ][
        "evidence_level"
    ]

    strict = metrics[
        "retrieval"
    ][
        "strict_page_level"
    ]

    print(
        "\nDOCUMENT-LEVEL RETRIEVAL"
    )

    print(
        f"Hit@1:      {document['hit_at_1']}"
    )
    print(
        f"Hit@3:      {document['hit_at_3']}"
    )
    print(
        f"Hit@5:      {document['hit_at_5']}"
    )
    print(
        f"MRR:        {document['mrr']}"
    )
    print(
        f"Precision@5:{document['precision_at_5']}"
    )
    print(
        f"Recall@5:   {document['recall_at_5']}"
    )

    print(
        "\nEVIDENCE-LEVEL RETRIEVAL"
    )

    print(
        f"Hit@1:      {evidence['hit_at_1']}"
    )
    print(
        f"Hit@3:      {evidence['hit_at_3']}"
    )
    print(
        f"Hit@5:      {evidence['hit_at_5']}"
    )
    print(
        f"MRR:        {evidence['mrr']}"
    )
    print(
        f"Precision@5:{evidence['precision_at_5']}"
    )

    print(
        "\nSTRICT PAGE DIAGNOSTIC"
    )

    print(
        f"Hit@1: {strict['hit_at_1']}"
    )
    print(
        f"Hit@3: {strict['hit_at_3']}"
    )
    print(
        f"Hit@5: {strict['hit_at_5']}"
    )
    print(
        f"MRR:   {strict['mrr']}"
    )

    print(
        "\nGENERATION"
    )

    print(
        "Answer accuracy:     "
        f"{metrics['generation']['answer_accuracy']}"
    )

    print(
        "Citation accuracy:   "
        f"{metrics['generation']['citation_accuracy']}"
    )

    print(
        "Abstention accuracy: "
        f"{metrics['generation']['abstention_accuracy']}"
    )

    print(
        "\nSAFETY"
    )

    print(
        "Pass rate: "
        f"{metrics['safety']['pass_rate']}"
    )

    print(
        "\nPERFORMANCE"
    )

    print(
        "Average latency: "
        f"{metrics['performance']['average_latency_seconds']} s"
    )

    print(
        "P50 latency:     "
        f"{metrics['performance']['p50_latency_seconds']} s"
    )

    print(
        "P95 latency:     "
        f"{metrics['performance']['p95_latency_seconds']} s"
    )


if __name__ == "__main__":

    configure_logging()


    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a RAG experiment "
            "against the fixed golden dataset."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to experiment configuration JSON."
        ),
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    )

    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT
            / config_path
        )

    summary = run_evaluation(
        config_path=config_path
    )

    print_metrics(
        summary
    )

# python evaluation/evaluate.py --config evaluation/configs/baseline_v1.json
# python evaluation/evaluate.py --config evaluation/configs/hybrid_v2.json