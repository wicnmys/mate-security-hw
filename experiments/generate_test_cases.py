#!/usr/bin/env python3
"""Generate test cases for SQL agent comparison experiments."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from agno.agent import Agent
from agno.models.anthropic import Claude
from pydantic import BaseModel, Field

from src.utils.schema_loader import load_schemas
from src.utils.validator import SQLValidator


class TestCase(BaseModel):
    """Single test case for agent evaluation."""
    question: str = Field(description="Natural language question")
    reference_sql: str = Field(description="Reference SQL query")
    reference_tables: list[str] = Field(description="Tables used in reference query")
    semantic_intent: str = Field(description="What the query is trying to achieve")
    category: str = Field(description="Category (endpoint, network, authentication, etc.)")


class TestCaseBatch(BaseModel):
    """Batch of generated test cases."""
    test_cases: list[TestCase] = Field(description="List of generated test cases")


class TestCaseGenerator:
    """Generate test cases using LLM judge."""

    def __init__(self, schema_path: str, model: str = "claude-sonnet-4-5"):
        """
        Initialize test case generator.

        Args:
            schema_path: Path to schema JSON file
            model: LLM model to use for generation
        """
        self.schemas = load_schemas(schema_path)
        self.validator = SQLValidator(self.schemas)
        self.model_name = model

        # Create agent for test case generation with caching enabled
        self.agent = Agent(
            name="test_case_generator",
            model=Claude(
                id=model,
                cache_system_prompt=True,  # Cache generation instructions
                cache_tool_definitions=True,  # Cache output schema
                cache_ttl="1h"  # Cache for 1 hour
            ),
            instructions=self._get_generator_instructions(),
            output_schema=TestCaseBatch,
            markdown=False
        )

    def _get_generator_instructions(self) -> str:
        """Get instructions for test case generation."""
        return """You are an expert at creating realistic test cases for SQL query generation systems.

Your task is to generate diverse, realistic test cases that a security analyst might ask about a security events database.

**Guidelines:**
1. Questions should be natural and realistic (how analysts actually talk)
2. Cover different query complexities and patterns
3. Include appropriate time ranges, severity levels, and security concepts
4. SQL must be syntactically correct PostgreSQL
5. Be specific and concrete (avoid vague questions)

**Complexity Levels:**
- **Simple**: Single table, basic WHERE clause, no aggregations
  - Example: "Show me all high-severity endpoint events"
  - SQL: SELECT * FROM endpoint_events WHERE severity IN ('high', 'critical')

- **Medium**: Single table with aggregation, GROUP BY, ORDER BY, or LIMIT
  - Example: "Which users had the most failed login attempts?"
  - SQL: SELECT user_name, COUNT(*) as attempts FROM authentication_events WHERE status = 'failure' GROUP BY user_name ORDER BY attempts DESC

- **Complex**: Multi-table JOINs, subqueries, or cross-domain correlation
  - Example: "Find processes executed by users with failed authentication from known malicious IPs"
  - SQL: SELECT DISTINCT pe.* FROM process_execution pe JOIN authentication_events ae ON pe.user_name = ae.user_name WHERE ae.status = 'failure' AND ae.source_ip IN (SELECT source_ip FROM threat_intelligence WHERE indicator_type = 'ip')

**Important:**
- Always include the list of tables used in reference_tables
- Provide a semantic_intent describing what the query does
- Assign a category based on the primary table/domain

Generate realistic, diverse test cases that cover different security scenarios."""

    def _build_schema_summaries(self) -> str:
        """Build concise schema summaries for prompt."""
        summaries = []
        for table_name, schema in self.schemas.items():
            category = schema.get('category', 'unknown')
            description = schema.get('description', '')
            # Get first few field names
            fields = schema.get('fields', [])
            field_names = [f['name'] for f in fields[:5]]
            field_str = ', '.join(field_names)
            if len(fields) > 5:
                field_str += f", ... ({len(fields)} total fields)"

            summaries.append(f"- **{table_name}** ({category}): {description}\n  Fields: {field_str}")

        return "\n".join(summaries)

    def generate_batch(
        self,
        complexity: str,
        count: int,
        category: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of test cases with specified complexity.

        Args:
            complexity: 'simple', 'medium', or 'complex'
            count: Number of test cases to generate
            category: Optional category to focus on

        Returns:
            List of generated and validated test cases
        """
        # Build schema context
        schema_summaries = self._build_schema_summaries()

        # Build prompt
        category_constraint = f"\n- Focus primarily on the **{category}** category" if category else ""

        prompt = f"""Generate {count} realistic security analyst questions with reference SQL queries.

**Complexity**: {complexity}
**Requirements**:{category_constraint}

**Available Tables:**
{schema_summaries}

Generate {count} diverse test cases covering different security scenarios at {complexity} complexity level.
Return them in the structured format."""

        try:
            # Call LLM to generate test cases
            run_output = self.agent.run(prompt)
            batch = run_output.content

            # Validate and convert to dicts
            validated_cases = []

            for i, test_case in enumerate(batch.test_cases):
                # Validate SQL
                validation = self.validator.validate(test_case.reference_sql, strict=False)

                if not validation['valid']:
                    print(f"⚠️  Skipping invalid SQL for: {test_case.question}")
                    print(f"   Errors: {validation['errors']}")
                    continue

                # Convert to dict and add metadata
                case_dict = {
                    'id': f"test_{complexity[:3]}_{i+1:03d}",
                    'complexity': complexity,
                    'category': test_case.category,
                    'question': test_case.question,
                    'reference_sql': test_case.reference_sql,
                    'reference_tables': test_case.reference_tables,
                    'semantic_intent': test_case.semantic_intent
                }

                validated_cases.append(case_dict)

            if len(validated_cases) < count * 0.7:  # If we lost too many to validation
                print(f"⚠️  Warning: Only {len(validated_cases)}/{count} test cases passed validation for {complexity}")

            return validated_cases

        except Exception as e:
            print(f"❌ Error generating batch: {e}")
            return []

    def generate_all(
        self,
        simple: int = 10,
        medium: int = 10,
        complex: int = 5
    ) -> Dict[str, Any]:
        """
        Generate all test cases.

        Args:
            simple: Number of simple test cases
            medium: Number of medium test cases
            complex: Number of complex test cases

        Returns:
            Complete test suite with metadata
        """
        all_cases = []

        print(f"\n🔄 Generating {simple} simple test cases...")
        all_cases.extend(self.generate_batch("simple", simple))

        print(f"\n🔄 Generating {medium} medium test cases...")
        all_cases.extend(self.generate_batch("medium", medium))

        print(f"\n🔄 Generating {complex} complex test cases...")
        all_cases.extend(self.generate_batch("complex", complex))

        # Build test suite
        test_suite = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator_model": self.model_name,
                "total_cases": len(all_cases),
                "complexity_distribution": {
                    "simple": sum(1 for c in all_cases if c['complexity'] == 'simple'),
                    "medium": sum(1 for c in all_cases if c['complexity'] == 'medium'),
                    "complex": sum(1 for c in all_cases if c['complexity'] == 'complex')
                }
            },
            "test_cases": all_cases
        }

        return test_suite

    def save(self, test_suite: Dict[str, Any], output_path: str):
        """
        Save test cases to JSON file.

        Args:
            test_suite: Test suite dictionary
            output_path: Path to output file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(test_suite, f, indent=2)

        total = test_suite['metadata']['total_cases']
        dist = test_suite['metadata']['complexity_distribution']
        print(f"\n✅ Saved {total} test cases to {output_path}")
        print(f"   Distribution: {dist['simple']} simple, {dist['medium']} medium, {dist['complex']} complex")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate test cases for SQL agent experiments")
    parser.add_argument(
        "--schema-path",
        default="schemas/dataset.json",
        help="Path to schema file"
    )
    parser.add_argument(
        "--output",
        default="experiments/test_cases/generated_test_cases.json",
        help="Output path for test cases"
    )
    parser.add_argument(
        "--simple",
        type=int,
        default=10,
        help="Number of simple test cases"
    )
    parser.add_argument(
        "--medium",
        type=int,
        default=10,
        help="Number of medium test cases"
    )
    parser.add_argument(
        "--complex",
        type=int,
        default=5,
        help="Number of complex test cases"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5",
        help="LLM model for generation"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SQL AGENT TEST CASE GENERATOR")
    print("=" * 70)
    print(f"Schema: {args.schema_path}")
    print(f"Model: {args.model}")
    print(f"Target: {args.simple} simple + {args.medium} medium + {args.complex} complex")
    print("=" * 70)

    generator = TestCaseGenerator(args.schema_path, args.model)
    test_suite = generator.generate_all(args.simple, args.medium, args.complex)
    generator.save(test_suite, args.output)

    print("\n✅ Test case generation complete!")
    print(f"   Run experiments with: python experiments/run_experiments.py")


if __name__ == "__main__":
    main()
