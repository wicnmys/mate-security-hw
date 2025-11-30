#!/usr/bin/env python3
"""Test script to analyze table description token lengths."""

from transformers import AutoTokenizer
from src.utils.schema_loader import load_schemas, get_table_description
from src.utils.constants import DEFAULT_EMBEDDING_MODEL

def main():
    # Load schemas
    schemas = load_schemas("schemas/dataset.json")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{DEFAULT_EMBEDDING_MODEL}")

    print(f"Analyzing {len(schemas)} tables...")
    print("="*80)

    results = []

    for table_name, schema in schemas.items():
        # Get description with descriptions included
        desc_with = get_table_description(table_name, schema, include_descriptions=True)
        tokens_with = len(tokenizer.encode(desc_with))

        # Get description without descriptions
        desc_without = get_table_description(table_name, schema, include_descriptions=False)
        tokens_without = len(tokenizer.encode(desc_without))

        results.append({
            'table': table_name,
            'tokens_with_desc': tokens_with,
            'tokens_without_desc': tokens_without,
            'num_fields': len(schema.get('fields', []))
        })

        print(f"\n{table_name}:")
        print(f"  Fields: {len(schema.get('fields', []))}")
        print(f"  Tokens (with descriptions): {tokens_with}")
        print(f"  Tokens (without descriptions): {tokens_without}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    max_with = max(results, key=lambda x: x['tokens_with_desc'])
    max_without = max(results, key=lambda x: x['tokens_without_desc'])

    print(f"\nMaximum tokens WITH descriptions:")
    print(f"  Table: {max_with['table']}")
    print(f"  Tokens: {max_with['tokens_with_desc']}")
    print(f"  Fields: {max_with['num_fields']}")

    print(f"\nMaximum tokens WITHOUT descriptions:")
    print(f"  Table: {max_without['table']}")
    print(f"  Tokens: {max_without['tokens_without_desc']}")
    print(f"  Fields: {max_without['num_fields']}")

    avg_with = sum(r['tokens_with_desc'] for r in results) / len(results)
    avg_without = sum(r['tokens_without_desc'] for r in results) / len(results)

    print(f"\nAverage tokens WITH descriptions: {avg_with:.1f}")
    print(f"Average tokens WITHOUT descriptions: {avg_without:.1f}")

    print(f"\nTotal tokens if embedding all tables:")
    print(f"  WITH descriptions: {sum(r['tokens_with_desc'] for r in results)}")
    print(f"  WITHOUT descriptions: {sum(r['tokens_without_desc'] for r in results)}")

if __name__ == "__main__":
    main()
