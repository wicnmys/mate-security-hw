#!/usr/bin/env python3
"""
Main CLI entrypoint for SQL Query Agent.

Usage:
    python main.py "Show me high-severity events from the last 24 hours"
    python main.py "Which users had failed login attempts?" --explain
    python main.py "Find suspicious file access events" --model gpt-4
"""

import argparse
import json
import sys
from pathlib import Path

from src.agents.semantic_agent import SemanticAgent


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="SQL Query Agent - Convert natural language to SQL queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Show me all high-severity security events from the last 24 hours"
  python main.py "Which users had the most failed login attempts?" --explain
  python main.py "Find suspicious file access events" --model gpt-4
  python main.py "What are the top 10 most common security event types?" --json
        """
    )

    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to convert to SQL"
    )

    parser.add_argument(
        "--schema-path",
        type=str,
        default="schemas/dataset.json",
        help="Path to schema JSON file (default: schemas/dataset.json)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="LLM model to use (default: claude-3-5-sonnet-20241022)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of tables to retrieve for context (default: 5)"
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show which tables were retrieved and why"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )

    args = parser.parse_args()

    # Validate schema path
    if not Path(args.schema_path).exists():
        print(f"Error: Schema file not found: {args.schema_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize agent
        print("Initializing agent...", file=sys.stderr)
        agent = SemanticAgent(
            schema_path=args.schema_path,
            model=args.model,
            top_k_tables=args.top_k
        )

        # Show retrieval explanation if requested
        if args.explain:
            print("\n" + "="*80, file=sys.stderr)
            print("RETRIEVAL EXPLANATION", file=sys.stderr)
            print("="*80, file=sys.stderr)

            retrieval_info = agent.explain_retrieval(args.question, k=args.top_k)

            print(f"\nQuestion: {retrieval_info['question']}\n", file=sys.stderr)
            print(f"Retrieved {len(retrieval_info['tables_retrieved'])} tables:\n", file=sys.stderr)

            for i, table in enumerate(retrieval_info['tables_retrieved'], 1):
                print(f"{i}. {table['table']} (score: {table['score']:.3f})", file=sys.stderr)
                print(f"   Category: {table['category']}", file=sys.stderr)
                print(f"   Description: {table['description']}\n", file=sys.stderr)

            print("="*80 + "\n", file=sys.stderr)

        # Generate SQL query
        print("Generating SQL query...\n", file=sys.stderr)
        result = agent.run(args.question)

        # Output results
        if args.json:
            # JSON output to stdout
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            print("="*80)
            print("QUESTION")
            print("="*80)
            print(args.question)
            print()

            if result.get('error'):
                print("="*80)
                print("ERROR")
                print("="*80)
                print(result['error'])
                sys.exit(1)

            print("="*80)
            print("SQL QUERY")
            print("="*80)
            print(result['query'])
            print()

            print("="*80)
            print("EXPLANATION")
            print("="*80)
            print(result['explanation'])
            print()

            print("="*80)
            print("METADATA")
            print("="*80)
            print(f"Tables Used: {', '.join(result['tables_used'])}")
            print(f"Confidence: {result['confidence']:.2f}")

            if result.get('reasoning_steps'):
                print("\nReasoning Steps:")
                for step in result['reasoning_steps']:
                    print(f"  - {step}")

            print()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        if args.json:
            print(json.dumps({'error': str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
