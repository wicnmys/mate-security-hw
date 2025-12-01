# Experimental Framework

This directory contains the experimental framework for comparing SQL agent architectures.

## Overview

The framework enables systematic comparison of different SQL agent variants on:
- **Correctness**: Semantic evaluation by LLM judge (0.0-1.0)
- **Latency**: End-to-end query generation time (milliseconds)
- **Token Usage**: Estimated tokens for cost comparison
- **Retrieval Precision**: % of retrieved tables actually used in SQL

## Quick Start

### 1. Generate Test Cases

Generate synthetic test cases using LLM (run once, reuse results):

```bash
python experiments/generate_test_cases.py \
  --schema-path schemas/dataset.json \
  --output experiments/test_cases/generated_test_cases.json \
  --simple 10 \
  --medium 10 \
  --complex 5
```

**Options:**
- `--simple N`: Number of simple test cases (single table queries)
- `--medium N`: Number of medium test cases (aggregations, GROUP BY)
- `--complex N`: Number of complex test cases (JOINs, subqueries)
- `--model`: LLM model for generation (default: `claude-sonnet-4-5`)

**Output:** JSON file with test cases (question, reference SQL, expected tables, complexity, category)

### 2. Run Experiments

Compare agents on the generated test cases:

```bash
python experiments/run_experiments.py \
  --test-cases experiments/test_cases/generated_test_cases.json \
  --schema-path schemas/dataset.json \
  --agents keyword semantic \
  --output experiments/results/experiment_results.json \
  --top-k 5
```

**Options:**
- `--agents`: Which agents to test (choices: `keyword`, `semantic`)
- `--top-k N`: Number of tables to retrieve per query
- `--judge-model`: LLM model for correctness evaluation (default: `claude-sonnet-4-5`)

**Output:** JSON file with detailed results and aggregate metrics

### 3. Generate Report

Create markdown comparison report:

```bash
python experiments/generate_report.py \
  --results experiments/results/experiment_results.json \
  --output experiments/comparison.md
```

**Output:** Markdown report with:
- Executive summary
- Methodology
- Overall results table
- Breakdown by complexity and category
- Failure analysis
- Key insights and recommendations

## Complete Pipeline

Run all steps in sequence:

```bash
# Step 1: Generate test cases (run once)
python experiments/generate_test_cases.py --simple 10 --medium 10 --complex 5

# Step 2: Run experiments
python experiments/run_experiments.py --agents keyword semantic

# Step 3: Generate report
python experiments/generate_report.py

# View report
cat experiments/comparison.md
```

## Architecture

### Components

1. **Test Case Generator** (`generate_test_cases.py`)
   - Uses Agno Agent with Claude to generate questions + reference SQL
   - Validates SQL before accepting (uses SQLValidator)
   - Saves to JSON for reuse across experiments

2. **Experiment Runner** (`run_experiments.py`)
   - Accepts agents dynamically (not hardcoded)
   - Runs each test case through each agent
   - Measures latency, tokens, retrieval precision
   - Uses LLM judge to evaluate correctness

3. **Report Generator** (`generate_report.py`)
   - Loads experiment results JSON
   - Generates comprehensive markdown report
   - Includes statistical analysis and recommendations

4. **Utilities** (`utils/`)
   - `llm_judge.py`: LLM-as-judge for semantic correctness evaluation
   - `metrics.py`: Retrieval precision, token counting, aggregation

### Design Principles

- **Generalized**: Experiment runner accepts any agents via dependency injection
- **Reproducible**: Test cases stored in JSON, can be rerun cheaply
- **Extensible**: Add new agents by inheriting from `BaseAgent`
- **Transparent**: Detailed results saved for analysis

## Adding New Agents

To compare a new agent architecture:

1. Implement agent inheriting from `BaseAgent`:
   ```python
   from src.agents.base import BaseAgent

   class MyAgent(BaseAgent):
       def run(self, question: str) -> Dict[str, Any]:
           # Your implementation
           pass
   ```

2. Modify `run_experiments.py` to support your agent:
   ```python
   if "myagent" in args.agents:
       from src.agents.my_agent import MyAgent
       agents['myagent'] = MyAgent(schema_path=args.schema_path)
   ```

3. Run experiments:
   ```bash
   python experiments/run_experiments.py --agents keyword semantic myagent
   ```

## Cost Estimates

Based on Claude Sonnet 4.5 pricing ($3/MTok input, $15/MTok output):

- **Test case generation**: ~25 test cases × 2K tokens = ~$0.15
- **Experiment run**: ~25 test cases × 2 agents × 3K tokens = ~$0.45
- **Correctness evaluation**: ~25 test cases × 1K tokens = ~$0.08
- **Total**: ~$0.70 per complete run

Note: Test cases are generated once and reused, so subsequent runs only cost ~$0.53.

## Output Structure

### Test Cases JSON
```json
{
  "metadata": {
    "generated_at": "2025-11-30T...",
    "generator_model": "claude-sonnet-4-5",
    "total_cases": 25,
    "complexity_distribution": {"simple": 10, "medium": 10, "complex": 5}
  },
  "test_cases": [
    {
      "id": "test_sim_001",
      "complexity": "simple",
      "category": "endpoint",
      "question": "Show me all high-severity endpoint events",
      "reference_sql": "SELECT * FROM endpoint_events WHERE severity IN ('high', 'critical')",
      "reference_tables": ["endpoint_events"],
      "semantic_intent": "Retrieve high-severity security events from endpoints"
    }
  ]
}
```

### Experiment Results JSON
```json
{
  "metadata": {
    "timestamp": "2025-11-30T...",
    "agents": ["keyword", "semantic"],
    "total_test_cases": 25,
    "judge_model": "claude-sonnet-4-5"
  },
  "results": [
    {
      "agent": "keyword",
      "test_case_id": "test_sim_001",
      "question": "...",
      "reference_sql": "...",
      "generated_sql": "...",
      "correctness_score": 0.95,
      "correctness_reasoning": "...",
      "correctness_issues": [],
      "latency_ms": 1250.5,
      "total_tokens": 2500,
      "retrieval_precision": 1.0,
      "retrieved_tables": ["endpoint_events"],
      "reference_tables": ["endpoint_events"],
      "complexity": "simple",
      "category": "endpoint",
      "confidence": 0.9
    }
  ],
  "summary": {
    "keyword": {
      "overall": {
        "avg_correctness": 0.85,
        "avg_latency_ms": 1200,
        "avg_total_tokens": 2400,
        "avg_retrieval_precision": 0.88
      },
      "by_complexity": { ... },
      "by_category": { ... }
    }
  }
}
```

## Troubleshooting

### "Module not found" errors
Ensure you've installed dependencies:
```bash
pip install -r requirements-dev.txt
```

### "No API key" errors
Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Test case generation fails
- Check schema file exists: `schemas/dataset.json`
- Verify schema format (see `src/utils/schema_loader.py`)
- Try reducing test case count if hitting rate limits

### Experiment runner hangs
- Check agent initialization (may download embeddings first time)
- Verify LLM API access
- Try running with single agent first: `--agents keyword`

## Next Steps

After reviewing experiment results:
1. Identify best-performing agent for production
2. Analyze failure cases to improve prompts
3. Consider hybrid approaches (combine retrieval strategies)
4. Add more test cases for specific edge cases
5. Implement recommended improvements from report
