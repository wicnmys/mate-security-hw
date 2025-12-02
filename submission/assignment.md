# Mate Security - Home Assessment

## Overview

Build an intelligent SQL query generation agent that can understand natural language questions and generate appropriate SQL queries for a large security events database.

## Context

You have access to a security events database with the following characteristics:

- **Scale:** ~1,000 tables
- **Complexity:** Up to 300 fields per table
- **Data Structure:** Each table has a name and schema definition

## Task

Create an agent using the **agno framework** ([link](https://github.com/agno-agi/agno)) that takes natural language questions as input and suggests appropriate SQL queries to answer those questions.

## Requirements

### Core Functionality

1. **Schema Understanding**
   - Parse and understand the database schema structure
   - Handle 1k tables with varying complexity
   - Process tables with up to 300 fields each

2. **Natural Language Processing**
   - Accept questions in natural language
   - Extract intent and entities from user queries
   - Handle ambiguous or incomplete questions

3. **SQL Query Generation**
   - Generate syntactically correct SQL queries
   - Select appropriate tables and fields
   - Handle JOIN operations when multiple tables are needed
   - Apply appropriate filters and conditions

4. **Agent Implementation**
   - Use the agno framework for agent orchestration
   - Implement proper reasoning and decision-making logic
   - Handle edge cases and errors gracefully

## Expected Deliverables

1. **Source Code**
   - Well-structured, readable Python code
   - Proper use of the agno framework
   - Modular design with clear separation of concerns

2. **Documentation**
   - README with setup instructions
   - Architecture explanation
   - Example usage with sample queries

3. **Test Cases**
   - At least 5 example questions with expected SQL outputs
   - Edge case handling demonstrations
   - Unit tests for critical components

## Technical Specifications

### Input Dataset Format

```python
# Schema structure example
schema = {
    "table_name": "security_events",
    "fields": [
        {"name": "event_id", "type": "INTEGER", "description": "Unique event identifier"},
        {"name": "timestamp", "type": "TIMESTAMP", "description": "Event occurrence time"},
        {"name": "severity", "type": "STRING", "description": "Event severity level"},
        # ... more fields
    ]
}
```

### Input Format

```json
{
    "question": "Give me the recent top 100 security events with high-sevirity"
}
```

### Output Format

```json
{
    "query": "SELECT event_id, timestamp, severity FROM security_events WHERE severity = 'HIGH' ORDER BY timestamp DESC LIMIT 100",
    "explanation": "This query retrieves the most recent 100 high-severity security events",
    "tables_used": ["security_events"],
    "confidence": 0.95
}
```

## Evaluation Criteria

### Technical Excellence (30%)
- Correct implementation of agno framework patterns
- Code quality, structure, and best practices
- Efficient schema search and query generation
- Error handling and edge case management

### SQL Query Quality (30%)
- Correctness of generated SQL syntax
- Appropriate table and field selection
- Optimization considerations
- Handling of complex queries (JOINs, aggregations, subqueries)

### Agent Design (30%)
- Reasoning transparency
- Context management
- Ability to handle ambiguous queries
- Scalability considerations for large schemas

### Documentation & Testing (10%)
- Clear setup and usage instructions
- Comprehensive test coverage
- Example demonstrations
- Architecture documentation

## Sample Test Questions

Your solution should handle questions like:

1. "Show me all high-severity security events from the last 24 hours"
2. "Which users had the most failed login attempts this month?"
3. "Find all suspicious file access events related to sensitive documents"
4. "What are the top 10 most common security event types?"
5. "Show me events where the same IP address triggered multiple alerts"

## Bonus Points

- Implement query validation before execution
- Add support for query refinement based on feedback
- Include schema similarity search for relevant table discovery
- Implement query explanation in natural language
- Add support for multi-step reasoning for complex questions
- Include confidence scoring for generated queries

## Submission Guidelines

### What to Submit

1. Git repository (GitHub/GitLab) with:
   - Source code
   - Requirements.txt or pyproject.toml
   - README.md
   - Tests and examples

2. Brief video demo (5 minutes max) showing:
   - Setup process
   - Running sample queries
   - Explanation of your approach

### Time Expectation

- **Estimated time:** 4-6 hours

### Submission Format

- Share repository link via email
- Add any additional notes or assumptions in README

## Resources

- **Agno Framework:** https://github.com/agno-agi/agno
- **Sample schema files:** Attached to the email
- **SQL syntax reference:** PostgreSQL/MySQL

---

**Good luck! We're excited to see your solution.**
