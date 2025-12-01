# SQL Agent Comparison Report

## Executive Summary

**Experiment Date:** 2025-12-01T22:49:52.120801

**Agents Compared:** keyword, semantic, react

**Test Cases:** 63

**Evaluation Model:** claude-sonnet-4-5


**Winner:** REACT (Correctness: 67.6%)

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
- **ReAct Agent**: Iterative reasoning + acting with retrieval and validation tools

## Overall Results

| Agent | Correctness | Latency (ms) | Tokens | Retrieval Precision |
|-------|-------------|--------------|--------|---------------------|
| KEYWORD | 60.0% | 9916 | 195 | 80.2% |
| SEMANTIC | 63.3% | 10612 | 209 | 93.7% |
| REACT | 67.6% | 31597 | 204 | 91.3% |

## Results by Complexity

### Simple Queries

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 71.5% | 9498 | 171 | 85.0% |
| SEMANTIC | 75.5% | 10103 | 186 | 100.0% |
| REACT | 73.5% | 28499 | 147 | 95.0% |

### Medium Queries

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 49.4% | 9293 | 179 | 77.8% |
| SEMANTIC | 55.0% | 9982 | 181 | 88.9% |
| REACT | 68.3% | 26907 | 207 | 88.9% |

### Complex Queries

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 50.0% | 14804 | 383 | 66.7% |
| SEMANTIC | 40.0% | 15987 | 443 | 83.3% |
| REACT | 35.0% | 68192 | 478 | 83.3% |

## Results by Category

### Application

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 50.0% | 9200 | 157 | 50.0% |
| SEMANTIC | 80.0% | 9742 | 162 | 100.0% |
| REACT | 80.0% | 25789 | 127 | 100.0% |

### Authentication

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 80.0% | 8985 | 163 | 100.0% |
| SEMANTIC | 80.0% | 9258 | 180 | 100.0% |
| REACT | 80.0% | 26666 | 183 | 100.0% |

### Cloud

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 43.3% | 12285 | 288 | 88.9% |
| SEMANTIC | 40.0% | 12821 | 300 | 100.0% |
| REACT | 40.0% | 47187 | 302 | 100.0% |

### Dlp

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 50.0% | 9910 | 208 | 100.0% |
| SEMANTIC | 45.0% | 10622 | 208 | 100.0% |
| REACT | 50.0% | 30103 | 204 | 100.0% |

### Email

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 77.5% | 9925 | 204 | 100.0% |
| SEMANTIC | 70.0% | 9505 | 202 | 100.0% |
| REACT | 80.0% | 24275 | 196 | 100.0% |

### Endpoint

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 80.0% | 9760 | 194 | 91.7% |
| SEMANTIC | 72.5% | 10963 | 212 | 91.7% |
| REACT | 77.5% | 31809 | 218 | 91.7% |

### Network

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 13.3% | 8769 | 138 | 33.3% |
| SEMANTIC | 36.7% | 9535 | 152 | 66.7% |
| REACT | 70.0% | 24581 | 137 | 66.7% |

### Security_Ops

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 85.0% | 9588 | 164 | 100.0% |
| SEMANTIC | 85.0% | 9805 | 221 | 100.0% |
| REACT | 30.0% | 36310 | 130 | 100.0% |

### Vulnerability

| Agent | Correctness | Latency (ms) | Tokens | Precision |
|-------|-------------|--------------|--------|-----------|
| KEYWORD | 85.0% | 10198 | 204 | 75.0% |
| SEMANTIC | 87.5% | 11933 | 227 | 100.0% |
| REACT | 85.0% | 35508 | 274 | 75.0% |

## Integrity Score Results

*No integrity test cases in results.*

## Failure Analysis

### KEYWORD Failures (7 cases < 0.5)

**Common Issues:**
- Queries wrong table: 'network_traffic' instead of 'firewall_events' (2 cases)
- Uses incorrect action value: 'blocked' instead of 'deny' (1 cases)
- Will not return firewall events at all, making it unable to answer the question (1 cases)
- Wrong data_type filter value: uses 'PCI' instead of 'credit_card' (1 cases)
- Missing OR condition with content_matches LIKE '%credit card%' (1 cases)

**Example Failures:**

**Q:** What are the top 10 most queried domains in the last hour?
**Score:** 0.00
**Issue:** The generated SQL query uses the wrong column name 'query_name' instead of 'queried_domain'. This is a critical error because it selects and groups by an entirely different column than what the question asks for. While the overall structure (time filtering, aggregation, ordering, and limiting) is correct, querying the wrong column means this query will not answer the question 'What are the top 10 most queried domains in the last hour?' at all. It would return results for a different column that ...


**Q:** Find all blocked firewall events
**Score:** 0.20
**Issue:** The generated SQL query has fundamental errors that make it unable to answer the question correctly. While it attempts to filter for blocked events, it queries the wrong table ('network_traffic' instead of 'firewall_events') and uses an incorrect filter value ('blocked' instead of 'deny'). The question specifically asks for 'blocked firewall events', and the reference clearly indicates these are stored in the 'firewall_events' table with action = 'deny'. Querying a different table entirely means...


**Q:** Show me all blocked outbound connections in the last 24 hours, sorted by source IP
**Score:** 0.20
**Issue:** The generated SQL query has critical fundamental errors that prevent it from correctly answering the question. While it attempts to filter for outbound connections in the last 24 hours and sorts by source_ip, it fails on multiple essential criteria: (1) It queries the wrong table ('network_traffic' instead of 'firewall_events'), (2) It uses the wrong action filter value ('blocked' instead of 'deny'), and (3) It selects specific columns rather than all columns as requested by 'SELECT *'. The quer...

### SEMANTIC Failures (6 cases < 0.5)

**Common Issues:**
- Filters by data_type = 'PCI' instead of 'credit_card' - may not capture all credit card violations (1 cases)
- Missing the alternative filter condition: content_matches LIKE '%credit card%' (1 cases)
- Excludes important columns like data_type and content_matches that would help verify the match (1 cases)
- Uses SELECT with explicit columns instead of SELECT * - misses columns present in reference (1 cases)
- Queries wrong table: 'network_traffic' instead of 'firewall_events' (1 cases)

**Example Failures:**

**Q:** What are the top 10 most queried domains in the last hour?
**Score:** 0.00
**Issue:** The generated SQL has a critical semantic error: it uses the wrong column name 'query_name' instead of 'queried_domain'. While the overall query structure is correct (proper time filtering, aggregation, ordering, and limit), selecting and grouping by a different column means it answers a fundamentally different question. If 'query_name' and 'queried_domain' are different columns in the schema, this query returns completely different results than requested. The question specifically asks for 'dom...


**Q:** Show me all blocked outbound connections in the last 24 hours, sorted by source IP
**Score:** 0.20
**Issue:** The generated SQL query has critical fundamental errors that make it incorrect for answering the question. Most importantly, it queries the wrong table ('network_traffic' instead of 'firewall_events'), which is a major semantic error. Additionally, it uses an incorrect filter value ('blocked' instead of 'deny' for the action field). While the query demonstrates understanding of some requirements (outbound direction, 24-hour time window, sorting by source_ip), these correct elements cannot compen...


**Q:** Find all DLP violations involving credit card data
**Score:** 0.30
**Issue:** The generated query queries the correct table (dlp_events) and attempts to filter for credit card-related violations, which shows basic understanding. However, it has several significant issues: (1) It filters by data_type = 'PCI' instead of 'credit_card', which assumes PCI compliance data equates to credit card data - this may miss violations explicitly tagged as 'credit_card' and could include non-credit-card PCI data. (2) It completely omits the alternative filter condition 'content_matches L...

### REACT Failures (6 cases < 0.5)

**Common Issues:**
- Status filter uses wrong values: 'new' and 'investigating' instead of 'open' (1 cases)
- The IN clause with multiple values doesn't match the single 'open' status specified in the reference query (1 cases)
- Query would return records with different status values than what the question asks for (1 cases)
- Uses 'data_type = PCI' instead of 'data_type = credit_card' - different filtering value that may not match the same records (1 cases)
- Missing the 'content_matches LIKE %credit card%' condition entirely, which could identify violations through text pattern matching (1 cases)

**Example Failures:**

**Q:** List all high severity security alerts that are still open
**Score:** 0.30
**Issue:** The generated query correctly identifies the table (security_alerts) and partially addresses the severity requirement (severity = 'high'). However, it has a critical logic error in the status filter. The reference SQL looks for status = 'open', but the generated SQL searches for status IN ('new', 'investigating'). These are fundamentally different status values. While 'new' and 'investigating' might represent open states in some systems, the question explicitly asks for alerts that are 'still op...


**Q:** Show me all blocked outbound connections in the last 24 hours, sorted by source IP
**Score:** 0.30
**Issue:** The generated SQL has the correct overall structure and logic flow (filtering by action, direction, time, and ordering), but it queries the wrong table and uses an incorrect action value. The reference uses 'firewall_events' table with action='deny', while the generated query uses 'network_traffic' table with action='blocked'. This is a fundamental error because different tables likely have different schemas and data. The action value mismatch ('deny' vs 'blocked') suggests the generated query m...


**Q:** Show all cloud resource changes made by users in production environments today
**Score:** 0.30
**Issue:** The generated query has several critical issues that prevent it from correctly answering the question. While it queries the correct table and includes timestamp filtering for 'today', it fails on multiple key requirements: (1) It does not filter for user-made changes (missing the 'user_name IS NOT NULL' condition or equivalent 'changed_by IS NOT NULL'), (2) It incorrectly identifies production environments by searching for 'production' in tags using LIKE patterns instead of using the 'environmen...

## Key Insights

2. **KEYWORD is faster** by 696ms on average
3. **SEMANTIC has better retrieval** by 13.5 percentage points

### Performance by Complexity
- **Medium:** SEMANTIC performs better (5.6pp advantage)
- **Complex:** KEYWORD performs better (10.0pp advantage)

## Recommendations

### Production Deployment
**Recommended Agent:** REACT

- **Correctness:** 67.6%
- **Latency:** 31597ms
- **Retrieval Precision:** 91.3%

---

*Generated on 2025-12-02 at 00:37:34*