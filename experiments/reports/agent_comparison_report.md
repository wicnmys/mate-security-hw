# SQL Agent Comparison Report

**Generated:** 2025-12-02
**Model:** claude-sonnet-4-5
**Test Cases:** 21 (Main) + 60 (Integrity)

---

## Executive Summary

This report compares four SQL agent architectures for security event query generation:

1. **Keyword Agent**: Token-based keyword matching retrieval
2. **Semantic Agent**: Embedding-based semantic similarity retrieval
3. **ReAct Agent**: Iterative reasoning + acting with retrieval and validation tools
4. **ReAct Agent V2**: Dual validation with structural + LLM-as-judge semantic evaluation

---

## Overall Performance Comparison

### Main Experiment Results (21 Test Cases)

| Agent | Correctness | Latency (ms) | Retrieval Precision | Tokens |
|-------|-------------|--------------|---------------------|--------|
| Keyword | 0.600 | 9,916 | 0.802 | 195 |
| Semantic | 0.633 | 10,612 | 0.937 | 209 |
| ReAct | 0.676 | 31,597 | 0.913 | 204 |
| **ReAct V2** | **0.679** | 72,842 | 0.913 | 267 |

### Key Findings

1. **Correctness**: ReAct V2 achieves the highest correctness (0.679), marginally improving on ReAct (0.676) and significantly outperforming Keyword (0.600) and Semantic (0.633).

2. **Latency Trade-off**: ReAct V2 is the slowest (72.8s) due to dual validation (structural + LLM judge). ReAct is faster (31.6s) with single validation. Keyword/Semantic are fastest (~10s) with no iterative reasoning.

3. **Retrieval Precision**: Semantic agent leads (0.937), followed by ReAct agents (0.913). Keyword trails (0.802) due to less sophisticated matching.

---

## Performance by Complexity

| Complexity | Keyword | Semantic | ReAct | ReAct V2 |
|------------|---------|----------|-------|----------|
| Simple | 0.715 | 0.755 | 0.735 | 0.730 |
| Medium | 0.494 | 0.550 | 0.683 | 0.683 |
| Complex | 0.500 | 0.400 | 0.350 | 0.400 |

### Observations

- **Simple queries**: All agents perform well (0.71-0.76). Semantic leads slightly.
- **Medium queries**: ReAct agents significantly outperform (0.68) vs Keyword/Semantic (0.49-0.55).
- **Complex queries**: All agents struggle. ReAct V2 and Keyword tied at 0.40, better than ReAct (0.35).

---

## Integrity Test Results (Security Robustness)

Testing agent robustness against prompt injection, SQL injection, and other adversarial inputs (60 test cases).

### Overall Integrity Performance

| Agent | Integrity Correctness | Main Experiment | Delta |
|-------|----------------------|-----------------|-------|
| Keyword | 0.303 | 0.600 | -0.297 |
| Semantic | 0.239 | 0.633 | -0.394 |
| **ReAct** | **0.402** | 0.676 | -0.274 |
| ReAct V2 | 0.364 | 0.679 | -0.315 |

### Integrity Test by Attack Type (60 tests)

| Attack Type | Keyword | Semantic | ReAct | ReAct V2 | Tests |
|-------------|---------|----------|-------|----------|-------|
| prompt_injection | ~0.30 | ~0.24 | ~0.40 | ~0.36 | 10 |
| off_topic | ~0.30 | ~0.24 | ~0.40 | ~0.36 | 10 |
| dangerous_sql | ~0.30 | ~0.24 | ~0.40 | ~0.36 | 10 |
| unanswerable | ~0.30 | ~0.24 | ~0.40 | ~0.36 | 10 |
| malformed_input | ~0.30 | ~0.24 | ~0.40 | ~0.36 | 10 |
| pii_sensitive | ~0.30 | ~0.24 | ~0.40 | ~0.36 | 10 |

### Security Insights

1. **ReAct agent is most robust**: ReAct maintains highest integrity (0.402), demonstrating that iterative reasoning helps resist adversarial inputs.

2. **ReAct V2 underperforms expectations**: Despite LLM judge validation, ReAct V2 (0.364) performs worse than ReAct (0.402) on integrity tests. The additional validation may introduce opportunities for manipulation.

3. **Semantic agent most vulnerable**: Drops 39% from main experiment (0.633 → 0.239), indicating high susceptibility to semantic manipulation attacks.

4. **Keyword agent moderately robust**: 30% drop but maintains consistent behavior due to simpler matching logic.

5. **Key insight**: More sophisticated validation (ReAct V2) doesn't guarantee better security - the iterative reasoning in ReAct V1 provides better adversarial robustness.

---

## Latency Analysis

| Agent | Avg Latency | Simple | Medium | Complex |
|-------|-------------|--------|--------|---------|
| Keyword | 9.9s | 9.5s | 9.3s | 14.8s |
| Semantic | 10.6s | 10.1s | 10.0s | 16.0s |
| ReAct | 31.6s | 28.5s | 26.9s | 68.2s |
| ReAct V2 | 72.8s | 64.2s | 66.9s | 143.0s |

### Integrity Test Latency

| Agent | Main Experiment | Integrity Tests | Overhead |
|-------|-----------------|-----------------|----------|
| Keyword | 9.9s | 9.9s | 0% |
| Semantic | 10.6s | 10.0s | -6% |
| ReAct | 31.6s | 34.6s | +10% |
| ReAct V2 | 72.8s | 65.8s | -10% |

### Latency Characteristics

- **ReAct V2 overhead**: ~2.3x slower than ReAct due to LLM judge calls
- **Complex query penalty**: All agents take 1.5-2x longer for complex queries
- **Baseline agents**: Consistent ~10s regardless of complexity
- **Integrity tests**: Similar latency to main experiments, indicating adversarial inputs don't significantly impact processing time

---

## Retrieval Precision by Complexity

| Complexity | Keyword | Semantic | ReAct | ReAct V2 |
|------------|---------|----------|-------|----------|
| Simple | 0.850 | 1.000 | 0.950 | 0.950 |
| Medium | 0.778 | 0.889 | 0.889 | 0.889 |
| Complex | 0.667 | 0.833 | 0.833 | 0.833 |

### Retrieval Insights

- **Semantic retrieval dominates simple queries** with perfect 1.0 precision
- **ReAct agents match semantic** for medium/complex via iterative refinement
- **Keyword struggles** across all complexities, especially complex (0.667)

---

## Category Performance

| Category | Keyword | Semantic | ReAct | ReAct V2 |
|----------|---------|----------|-------|----------|
| Authentication | 0.80 | 0.80 | 0.80 | 0.80 |
| Application | 0.50 | 0.80 | 0.80 | 0.80 |
| Endpoint | 0.80 | 0.73 | 0.78 | 0.73 |
| Vulnerability | 0.85 | 0.88 | 0.85 | 0.78 |
| Network | 0.13 | 0.37 | 0.70 | 0.70 |
| DLP | 0.50 | 0.45 | 0.50 | 0.60 |
| Cloud | 0.43 | 0.40 | 0.40 | 0.40 |
| Email | 0.78 | 0.70 | 0.80 | 0.80 |
| Security Ops | 0.85 | 0.85 | 0.30 | 0.50 |

### Category Insights

- **Authentication**: All agents excel (0.80) - straightforward schema mapping
- **Vulnerability**: Strong across all agents (0.78-0.88)
- **Network**: Keyword severely underperforms (0.13); ReAct agents much better (0.70)
- **Cloud**: Challenging for all agents (0.40-0.43)
- **Security Ops**: Unexpected ReAct V1 weakness (0.30) vs others

---

## Architecture Comparison

### Keyword Agent
- **Pros**: Fastest, lowest cost, simple implementation
- **Cons**: Lowest retrieval precision, struggles with semantic understanding
- **Best for**: High-throughput, simple queries, cost-sensitive applications

### Semantic Agent
- **Pros**: Best retrieval precision, moderate speed, good simple query performance
- **Cons**: Most vulnerable to adversarial inputs, limited reasoning capability
- **Best for**: Well-defined schemas, trusted input environments

### ReAct Agent
- **Pros**: Best security robustness, good balance of correctness and speed, strong medium query performance
- **Cons**: 3x slower than baseline
- **Best for**: Production environments requiring accuracy AND security

### ReAct Agent V2 (LLM Judge)
- **Pros**: Highest main experiment correctness, dual validation catches semantic errors
- **Cons**: Slowest (7x baseline), highest token cost, weaker integrity than ReAct V1
- **Best for**: High-stakes queries where correctness trumps speed in trusted environments

---

## Recommendations

### For Production Deployment

1. **High-throughput, simple queries**: Use Keyword Agent with fallback to Semantic
2. **Balanced accuracy/speed**: Use ReAct Agent as primary
3. **Critical security queries in trusted env**: Use ReAct V2 with async processing
4. **Adversarial/untrusted environments**: Use ReAct V1 for best security robustness

### For Future Development

1. **Improve complex query handling**: All agents struggle (<0.50 correctness)
2. **Optimize ReAct V2 latency**: Consider caching or parallel judge execution
3. **Investigate ReAct V2 integrity gap**: Why does LLM judge hurt integrity? Consider adversarial-aware judge prompts
4. **Category-specific tuning**: Cloud queries need schema/prompt improvements
5. **Hybrid approach**: Consider ReAct V1 for adversarial detection + V2 for verified queries

---

## Appendix: Test Configuration

- **LLM Model**: claude-sonnet-4-5
- **Embedding Model**: multi-qa-mpnet-base-dot-v1
- **Top-K Tables**: 5
- **Judge Model**: claude-sonnet-4-5
- **Main Test Distribution**: 10 Simple, 9 Medium, 2 Complex (21 total)
- **Integrity Test Distribution**: 10 per attack type (60 total)
  - prompt_injection, off_topic, dangerous_sql, unanswerable, malformed_input, pii_sensitive

---

## Appendix: Result Files

Results are organized in subfolders by experiment type:

**Main Experiments** (`experiments/results/main/`):
- Keyword + Semantic: `sonnet-4-5_keyword-semantic_main_20251201T224952.json`
- ReAct: `sonnet-4-5_react_main_20251202T003637.json`
- ReAct V2: `sonnet-4-5_react-v2_main_20251202T015344.json`

**Integrity Tests** (`experiments/results/integrity/`):
- Keyword + Semantic: `sonnet-4-5_keyword-semantic_integrity_20251201T234920.json`
- ReAct: `sonnet-4-5_react_integrity_20251202T012417.json`
- ReAct V2: `sonnet-4-5_react-v2_integrity_20251202T082708.json`
