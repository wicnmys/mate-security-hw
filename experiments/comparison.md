# SQL Agent Comparison Report

## Executive Summary

**Experiment Date:** 2025-12-01T15:35:04.125699

**Agents Compared:** keyword, semantic

**Test Cases:** 21

**Evaluation Model:** claude-sonnet-4-5


**Winner:** SEMANTIC (Correctness: 62.9%)

## Methodology

### Test Case Generation
- Generated synthetic test cases using LLM (Claude Sonnet 4.5)
- Three complexity levels: simple (single table), medium (aggregations), complex (joins/subqueries)
- Each test case includes: question, reference SQL, expected tables, semantic intent

### Evaluation Metrics
1. **Correctness**: LLM-as-judge semantic evaluation (0.0-1.0 score)
2. **Latency**: End-to-end query generation time (milliseconds)
3. **Token Usage**: Estimated tokens for cost comparison
4. **Retrieval Precision**: % of retrieved tables actually used in SQL

### Agent Architectures
- **Keyword Agent**: Token-based keyword matching retrieval
- **Semantic Agent**: Embedding-based semantic similarity retrieval

## Overall Results

| Agent | Correctness | Latency (ms) | Tokens | Retrieval Precision |
|-------|-------------|--------------|--------|---------------------|
| KEYWORD | 58.8% | 10264 | 200 | 79.4% |
| SEMANTIC | 62.9% | 10717 | 203 | 87.6% |

## Results by Complexity

### Simple Queries

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 69.5% | 9587 | 176 | 85.0% |
| SEMANTIC | 76.5% | 10157 | 178 | 95.0% |

### Medium Queries

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 47.8% | 9939 | 184 | 77.8% |
| SEMANTIC | 55.0% | 9971 | 181 | 88.9% |

### Complex Queries

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 55.0% | 15112 | 400 | 58.3% |
| SEMANTIC | 30.0% | 16876 | 426 | 45.0% |

## Results by Category

### Application

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 50.0% | 8604 | 152 | 50.0% |
| SEMANTIC | 80.0% | 8452 | 150 | 100.0% |

### Authentication

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 80.0% | 8737 | 182 | 100.0% |
| SEMANTIC | 80.0% | 8846 | 166 | 100.0% |

### Cloud

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 50.0% | 12935 | 289 | 83.3% |
| SEMANTIC | 40.0% | 13629 | 343 | 80.0% |

### Dlp

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 35.0% | 12138 | 231 | 100.0% |
| SEMANTIC | 45.0% | 11680 | 180 | 100.0% |

### Email

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 77.5% | 9626 | 206 | 100.0% |
| SEMANTIC | 75.0% | 10042 | 199 | 100.0% |

### Endpoint

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 81.2% | 9893 | 188 | 91.7% |
| SEMANTIC | 70.0% | 11134 | 181 | 87.5% |

### Network

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 13.3% | 8648 | 140 | 33.3% |
| SEMANTIC | 40.0% | 9399 | 155 | 66.7% |

### Security_Ops

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 70.0% | 9719 | 185 | 100.0% |
| SEMANTIC | 70.0% | 10781 | 219 | 100.0% |

### Vulnerability

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 82.5% | 11647 | 222 | 75.0% |
| SEMANTIC | 85.0% | 11308 | 218 | 75.0% |

## Failure Analysis

### KEYWORD Failures (8 cases < 0.5)

**Common Issues:**
- Queries wrong table: 'network_traffic' instead of 'firewall_events' (2 cases)
- Uses incorrect filter value: action = 'blocked' instead of action = 'deny' (1 cases)
- Would not return firewall events as requested in the question (1 cases)
- Uses 'PCI' instead of 'credit_card' as the data_type filter value, which may not match the actual credit card violations (1 cases)
- Missing the content_matches LIKE '%credit card%' condition that provides a secondary way to identify credit card data (1 cases)

**Example Failures:**

**Q:** What are the top 10 most queried domains in the last hour?
**Score:** 0.00
**Issue:** The generated SQL query has a critical error: it uses the wrong column name 'query_name' instead of 'queried_domain'. This is a fundamental mistake th...


**Q:** Find all blocked firewall events
**Score:** 0.20
**Issue:** The generated SQL query has critical errors that prevent it from correctly answering the question. While it attempts to filter for blocked events, it ...


**Q:** Show me all blocked outbound connections in the last 24 hours, sorted by source IP
**Score:** 0.20
**Issue:** The generated query has multiple critical issues that make it fundamentally incorrect. Most importantly, it queries the wrong table ('network_traffic'...

### SEMANTIC Failures (6 cases < 0.5)

**Common Issues:**
- Uses 'PCI' instead of 'credit_card' for data_type filter - different categorical value (1 cases)
- Missing the content_matches LIKE '%credit card%' condition entirely - loses a significant filtering dimension (1 cases)
- Adds ORDER BY timestamp DESC which is not in reference - suggests different query purpose (1 cases)
- Uses AND logic implicitly (single WHERE condition) vs OR logic in reference (1 cases)
- Queries wrong table: 'network_traffic' instead of 'firewall_events' (1 cases)

**Example Failures:**

**Q:** What are the top 10 most queried domains in the last hour?
**Score:** 0.00
**Issue:** The generated SQL uses a completely incorrect column name 'query_name' instead of 'queried_domain'. This is a critical error because it queries a diff...


**Q:** Show me all blocked outbound connections in the last 24 hours, sorted by source IP
**Score:** 0.30
**Issue:** The generated SQL has several critical issues that prevent it from correctly answering the question. Most importantly, it queries the wrong table ('ne...


**Q:** Show all cloud resource changes made by users in production environments today
**Score:** 0.30
**Issue:** The generated query has fundamental issues that make it incompatible with the reference query. While both attempt to filter for production environment...

## Key Insights

2. **KEYWORD is faster** by 453ms on average
3. **SEMANTIC has better retrieval** by 8.3 percentage points

### Performance by Complexity
- **Simple:** SEMANTIC performs better (7.0pp advantage)
- **Medium:** SEMANTIC performs better (7.2pp advantage)
- **Complex:** KEYWORD performs better (25.0pp advantage)

## Recommendations

### Production Deployment
**Recommended Agent:** SEMANTIC

- **Correctness:** 62.9%
- **Latency:** 10717ms
- **Retrieval Precision:** 87.6%

### Future Improvements
1. **Hybrid Retrieval**: Combine keyword and semantic approaches
2. **Query Refinement**: Add iterative self-correction based on validation
3. **Example-Based Learning**: Few-shot prompting with similar queries
4. **Contextual Ranking**: Re-rank retrieved tables based on query complexity
5. **Schema Optimization**: Add table relationship metadata for better JOIN handling

---

*Generated on 2025-12-01 at 15:36:48*