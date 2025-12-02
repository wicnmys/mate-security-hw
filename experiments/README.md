# Experimental Framework

This directory contains the experimental framework for comparing SQL agent architectures.

## Overview

The framework enables systematic comparison of different SQL agent variants on:
- **Correctness**: Semantic evaluation by LLM judge (multiple judge types available)
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
- `--integrity N`: Number of integrity test cases per category (6 categories total)
- `--model`: LLM model for generation (default: `claude-sonnet-4-5`)

**Output:** JSON file with test cases (question, reference SQL, expected tables, complexity, category)

### 2. Run Experiments

Compare agents on the generated test cases:

```bash
# Run main experiment with default settings
python experiments/run_experiments.py \
  --experiment-type main \
  --agents keyword semantic react react-v2

# Run integrity tests
python experiments/run_experiments.py \
  --experiment-type integrity \
  --agents keyword semantic

# Quick test with limited cases
python experiments/run_experiments.py \
  --experiment-type main \
  --agents keyword \
  --limit 5
```

**Options:**
- `--experiment-type`: Type of experiment (`main`, `integrity`, `consistency`)
- `--agents`: Which agents to test (`keyword`, `semantic`, `react`, `react-v2`)
- `--judge`: Judge type (`correctness`, `categorical`, `integrity`)
- `--judge-model`: LLM model for evaluation (default: `claude-sonnet-4-5`)
- `--limit N`: Run only first N test cases (for testing)
- `--top-k N`: Number of tables to retrieve per query
- `--output`: Custom output path (auto-generated if not specified)

**Output:** JSON file in `experiments/results/{experiment_type}/` with standardized naming

### 3. Re-judge Existing Results

Re-evaluate results with a different judge:

```bash
python experiments/rejudge.py \
  experiments/results/main/sonnet_4_5_keyword_correctness_20251202T101500.json \
  --judge categorical
```

**Options:**
- `--judge`: New judge type (`correctness`, `categorical`, `integrity`)
- `--judge-model`: LLM model for evaluation
- `--output`: Custom output path

### 4. Generate Report

Create markdown comparison report:

```bash
python experiments/generate_report.py \
  --results experiments/results/main/*.json \
  --output experiments/reports/comparison.md
```

**How it works:** Each judge class implements its own `generate_report_sections()` method that produces judge-specific analysis (e.g., correctness breakdowns, integrity pass rates). The report generator orchestrates this by:
1. Generating shared sections (executive summary, agent config, latency/token comparison)
2. Calling each judge's `generate_report_sections()` for specialized analysis
3. Stitching everything together into a unified markdown report

**Tip:** This decoupled design means you can change evaluation criteria without re-running experiments. Simply add a new judge class, use `rejudge.py` to re-evaluate existing results, then regenerate the report.

## Experiment Types

| Type | Description | Default Judge | Test Cases |
|------|-------------|---------------|------------|
| `main` | Standard SQL generation | `correctness` | `generated_test_cases.json` |
| `integrity` | Security/adversarial testing | `integrity` | `integrity_test_cases.json` |
| `consistency` | Cross-run consistency (future) | `correctness` | `consistency_test_cases.json` |

## Judge Types

### Correctness Judge (`correctness`)
- **Score**: Float 0.0-1.0
- **Use for**: General SQL quality evaluation
- **Rubric**:
  - 1.0: Perfectly correct
  - 0.9: Minor cosmetic differences
  - 0.8: Minor issues (missing ORDER BY)
  - 0.7: One significant issue
  - 0.5-0.6: Partially correct
  - 0.0-0.2: Completely wrong

### Categorical Judge (`categorical`)
- **Score**: Integer 1-5
- **Use for**: Discrete quality buckets
- **Categories**:
  - 5 (PERFECT): Semantically equivalent
  - 4 (GOOD): Functionally correct, minor issues
  - 3 (PARTIAL): Some correct, incomplete
  - 2 (POOR): Right tables, wrong logic
  - 1 (WRONG): Wrong tables or syntax errors

### Integrity Judge (`integrity`)
- **Score**: Boolean pass/fail + confidence
- **Use for**: Security/adversarial testing
- **Evaluates**: Prompt injection resistance, SQL injection handling, off-topic refusal

## Directory Structure

```
experiments/
├── configs/                    # Experiment configuration
│   ├── __init__.py
│   └── experiment_config.py    # ExperimentConfig dataclass
├── judges/                     # Judge implementations
│   ├── __init__.py
│   ├── base.py                 # BaseJudge abstract class
│   ├── correctness_judge.py    # 0.0-1.0 scoring
│   ├── categorical_judge.py    # 1-5 scoring
│   └── integrity_judge.py      # Pass/fail scoring
├── planning/                   # Planning documents
├── reports/                    # Generated reports
├── results/                    # Experiment results (by type)
│   ├── main/                   # Main experiment results
│   ├── integrity/              # Security test results
│   └── consistency/            # Consistency test results
├── test_cases/                 # Test case definitions
│   ├── generated_test_cases.json
│   └── integrity_test_cases.json
├── utils/                      # Shared utilities
│   ├── llm_judge.py           # Legacy judge (deprecated)
│   └── metrics.py             # Metrics calculation
├── generate_test_cases.py      # Test case generator
├── generate_report.py          # Report generator
├── rejudge.py                  # Re-evaluate with different judge
├── run_experiments.py          # Main experiment runner
└── README.md                   # This file
```

## Output Naming Convention

Results are automatically named with the pattern:
```
{model}_{agents}_{judge}_{timestamp}.json
```

Example: `sonnet_4_5_keyword-semantic_correctness_20251202T101500.json`

## Complete Pipeline

```bash
# Step 1: Generate test cases (run once)
python experiments/generate_test_cases.py --simple 10 --medium 10 --complex 5

# Step 2: Run main experiments
python experiments/run_experiments.py --experiment-type main --agents keyword semantic react react-v2

# Step 3: Run integrity tests
python experiments/run_experiments.py --experiment-type integrity --agents keyword semantic react react-v2

# Step 4: Re-judge with categorical judge for comparison
python experiments/rejudge.py experiments/results/main/latest.json --judge categorical

# Step 5: Generate report
python experiments/generate_report.py
```

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

2. Register in `src/agents/registry.py`:
   ```python
   from src.agents.my_agent import MyAgent

   AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
       "keyword": KeywordAgent,
       "semantic": SemanticAgent,
       "react": ReActAgent,
       "react-v2": ReActAgentV2,
       "myagent": MyAgent,  # Add here
   }
   ```

3. Run experiments:
   ```bash
   python experiments/run_experiments.py --agents keyword semantic myagent
   ```

## Adding New Judges

To add a new evaluation judge:

1. Inherit from `BaseJudge`:
   ```python
   from experiments.judges.base import BaseJudge

   class MyJudge(BaseJudge):
       judge_id = "myjudge_v1"

       def __init__(self, model: str = "claude-sonnet-4-5"):
           self.model_name = model
           # Initialize LLM agent...

       def evaluate(self, question, reference_sql, generated_sql, **kwargs):
           # Return evaluation dict
           return {'score': ..., 'reasoning': ..., 'issues': [...]}
   ```

2. Register in `run_experiments.py`:
   ```python
   JUDGE_REGISTRY = {
       "correctness": CorrectnessJudge,
       "categorical": CategoricalJudge,
       "integrity": IntegrityJudge,
       "myjudge": MyJudge,  # Add here
   }
   ```

## Output Format

### Experiment Results JSON
```json
{
  "metadata": {
    "timestamp": "2025-12-02T...",
    "experiment_type": "main",
    "agents": {
      "keyword": {"type": "keyword", "llm_model": "claude-sonnet-4-5", ...}
    },
    "total_test_cases": 21,
    "judge": {
      "type": "correctness_v1",
      "model": "claude-sonnet-4-5",
      "identifier": "claude-sonnet-4-5_correctness_v1"
    }
  },
  "results": [
    {
      "agent": "keyword",
      "test_case_id": "test_sim_001",
      "question": "...",
      "reference_sql": "...",
      "generated_sql": "...",
      "judge_type": "correctness_v1",
      "judge_identifier": "claude-sonnet-4-5_correctness_v1",
      "judge_evaluation": {
        "score": 0.95,
        "reasoning": "...",
        "issues": []
      },
      "correctness_score": 0.95,
      "latency_ms": 1250.5,
      "retrieval_precision": 1.0,
      "complexity": "simple",
      "category": "endpoint"
    }
  ],
  "summary": {
    "keyword": {
      "overall": {"avg_correctness": 0.85, "avg_latency_ms": 1200, ...},
      "by_complexity": {...},
      "by_category": {...}
    }
  }
}
```

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements-dev.txt
```

### "No API key" errors
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
- Use `--limit 5` to test with fewer cases

## Cost Estimates

Based on Claude Sonnet 4.5 pricing ($3/MTok input, $15/MTok output):

- **Test case generation**: ~25 test cases × 2K tokens = ~$0.15
- **Experiment run**: ~25 test cases × 2 agents × 3K tokens = ~$0.45
- **Judge evaluation**: ~25 test cases × 1K tokens = ~$0.08
- **Total**: ~$0.70 per complete run

Note: Test cases are generated once and reused, so subsequent runs only cost ~$0.53.
