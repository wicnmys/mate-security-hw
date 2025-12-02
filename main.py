#!/usr/bin/env python3
"""
Main CLI entrypoint for SQL Query Agent.

Usage:
    python main.py "Show me high-severity events from the last 24 hours"
    python main.py "Which users had failed login attempts?" --explain
    python main.py "Find suspicious file access events" --model gpt-4
"""

import os

# Suppress tokenizers parallelism warning (must be set before importing transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import readline  # Enable arrow keys and history in input()
import sys
from pathlib import Path

# Configure logging to suppress verbose error tracebacks in CLI output
logging.basicConfig(level=logging.WARNING)

from src.agents.registry import AGENT_REGISTRY, create_agent, get_agent_names


def run_session(agent, agent_name: str, output_json: bool = False):
    """Run an interactive session with the agent.

    Args:
        agent: Initialized agent instance
        agent_name: Name of the agent type for display
        output_json: Whether to output results as JSON
    """
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Interactive session started with {agent_name} agent", file=sys.stderr)
    print("Type 'exit' or 'quit' to end the session", file=sys.stderr)
    print("Type 'help' for available commands", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    while True:
        try:
            question = input("query> ").strip()

            if not question:
                continue

            if question.lower() in ('exit', 'quit', 'q'):
                print("\nSession ended.", file=sys.stderr)
                break

            if question.lower() == 'help':
                print("\nAvailable commands:", file=sys.stderr)
                print("  exit, quit, q  - End the session", file=sys.stderr)
                print("  help           - Show this help message", file=sys.stderr)
                print("  <question>     - Ask a natural language question\n", file=sys.stderr)
                continue

            # Process the query
            print("Generating SQL query...\n", file=sys.stderr)
            result = agent.run(question)

            if output_json:
                print(json.dumps(result, indent=2))
            else:
                if result.get('error'):
                    print(f"Error: {result['error']}\n")
                else:
                    print(f"SQL: {result['query']}")
                    print(f"Tables: {', '.join(result['tables_used'])}")
                    print(f"Confidence: {result['confidence']:.2f}")
                    print(f"Explanation: {result['explanation']}\n")

        except EOFError:
            print("\nSession ended.", file=sys.stderr)
            break
        except KeyboardInterrupt:
            print("\n", file=sys.stderr)
            continue  # Allow Ctrl+C to cancel current input without exiting


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="SQL Query Agent - Convert natural language to SQL queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Show me all high-severity security events from the last 24 hours"
  python main.py "Which users had the most failed login attempts?" --explain
  python main.py "Find suspicious file access events" --agent react
  python main.py "What are the top 10 most common security event types?" --json
  python main.py "Show network traffic anomalies" --agent react-v2
  python main.py --session --agent react  # Start interactive session
        """
    )

    parser.add_argument(
        "question",
        type=str,
        nargs='?',  # Make optional for session mode
        default=None,
        help="Natural language question to convert to SQL (optional in session mode)"
    )

    parser.add_argument(
        "--session",
        action="store_true",
        help="Start an interactive session (initialize once, query multiple times)"
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
        default="claude-sonnet-4-5",
        help="LLM model to use (default: claude-sonnet-4-5)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of tables to retrieve for context (default: 5). Only for keyword/semantic agents"
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show which tables were retrieved and why. Only for keyword/semantic agents"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="semantic",
        choices=get_agent_names(),
        help=f"Agent type to use (default: semantic). Options: {', '.join(get_agent_names())}"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.session and not args.question:
        parser.error("Either provide a question or use --session for interactive mode")

    # Validate schema path
    if not Path(args.schema_path).exists():
        print(f"Error: Schema file not found: {args.schema_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize agent
        print(f"Initializing {args.agent} agent...", file=sys.stderr)
        agent = create_agent(
            name=args.agent,
            schema_path=args.schema_path,
            model=args.model,
            top_k_tables=args.top_k
        )

        # Session mode: enter interactive loop
        if args.session:
            run_session(agent, args.agent, output_json=args.json)
            return

        # Show retrieval explanation if requested (only for agents that support it)
        if args.explain:
            if hasattr(agent, 'explain_retrieval'):
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
            else:
                print(f"\nNote: --explain is not supported for {args.agent} agent (retrieval happens dynamically)\n", file=sys.stderr)

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
