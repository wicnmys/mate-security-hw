"""Metric calculation utilities for agent evaluation."""

import re
from typing import List, Dict, Any


def calculate_retrieval_precision(retrieved: List[str], used: List[str]) -> float:
    """
    Calculate retrieval precision: % of retrieved tables actually used in SQL.

    Args:
        retrieved: List of table names retrieved by the agent
        used: List of table names actually used in the generated SQL

    Returns:
        Precision score between 0 and 1
    """
    if not retrieved:
        return 0.0

    retrieved_set = set(retrieved)
    used_set = set(used)
    intersection = retrieved_set & used_set

    return len(intersection) / len(retrieved_set)


def extract_tables_from_sql(sql: str) -> List[str]:
    """
    Extract table names from SQL query using regex.

    Args:
        sql: SQL query string

    Returns:
        List of table names found in the query
    """
    if not sql:
        return []

    # Pattern: FROM table_name or JOIN table_name
    pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
    matches = re.findall(pattern, sql, re.IGNORECASE)

    # Deduplicate and return
    return list(set(matches))


def calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate aggregate metrics across all test cases.

    Args:
        results: List of result dictionaries from experiment runs

    Returns:
        Dictionary of aggregate metrics
    """
    if not results:
        return {
            'avg_correctness': 0.0,
            'avg_latency_ms': 0.0,
            'avg_input_tokens': 0.0,
            'avg_output_tokens': 0.0,
            'avg_total_tokens': 0.0,
            'avg_retrieval_precision': 0.0,
            'syntax_valid_rate': 0.0
        }

    n = len(results)

    return {
        'avg_correctness': sum(r.get('correctness_score', 0.0) for r in results) / n,
        'avg_latency_ms': sum(r.get('latency_ms', 0.0) for r in results) / n,
        'avg_input_tokens': sum(r.get('input_tokens', 0) for r in results) / n,
        'avg_output_tokens': sum(r.get('output_tokens', 0) for r in results) / n,
        'avg_total_tokens': sum(r.get('total_tokens', 0) for r in results) / n,
        'avg_retrieval_precision': sum(r.get('retrieval_precision', 0.0) for r in results) / n,
        'syntax_valid_rate': sum(1 for r in results if r.get('validation', {}).get('valid', False)) / n
    }


def calculate_metrics_by_complexity(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics grouped by complexity level.

    Args:
        results: List of result dictionaries from experiment runs

    Returns:
        Dictionary mapping complexity levels to their metrics
    """
    complexity_groups = {}

    for result in results:
        complexity = result.get('complexity', 'unknown')
        if complexity not in complexity_groups:
            complexity_groups[complexity] = []
        complexity_groups[complexity].append(result)

    return {
        complexity: calculate_aggregate_metrics(group)
        for complexity, group in complexity_groups.items()
    }


def calculate_metrics_by_category(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics grouped by category.

    Args:
        results: List of result dictionaries from experiment runs

    Returns:
        Dictionary mapping categories to their metrics
    """
    category_groups = {}

    for result in results:
        category = result.get('category', 'unknown')
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(result)

    return {
        category: calculate_aggregate_metrics(group)
        for category, group in category_groups.items()
    }


def format_percentage(value: float) -> str:
    """Format a 0-1 value as a percentage string."""
    return f"{value * 100:.1f}%"


def format_metric(value: float, decimals: int = 2) -> str:
    """Format a metric value with specified decimal places."""
    return f"{value:.{decimals}f}"
