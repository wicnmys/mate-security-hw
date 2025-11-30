"""Schema loader for security events database."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoTokenizer


def load_schemas(schema_path: str) -> Dict[str, Any]:
    """
    Load database schemas from JSON file or directory.

    Args:
        schema_path: Path to schema JSON file or directory containing schema files

    Returns:
        Dictionary mapping table names to their schema definitions

    Raises:
        ValueError: If schema file is malformed or doesn't exist
    """
    path = Path(schema_path)

    if not path.exists():
        raise ValueError(f"Schema path does not exist: {schema_path}")

    # Handle single JSON file
    if path.is_file():
        if path.suffix != '.json':
            raise ValueError(f"Schema file must be JSON: {schema_path}")

        with open(path, 'r') as f:
            schemas = json.load(f)

        # Validate schema structure
        _validate_schemas(schemas)
        return schemas

    # Handle directory of JSON files
    if path.is_dir():
        schemas = {}
        json_files = list(path.glob('*.json'))

        if not json_files:
            raise ValueError(f"No JSON files found in directory: {schema_path}")

        for json_file in json_files:
            with open(json_file, 'r') as f:
                file_schemas = json.load(f)
                _validate_schemas(file_schemas)
                schemas.update(file_schemas)

        return schemas

    raise ValueError(f"Invalid schema path: {schema_path}")


def _validate_schemas(schemas: Dict[str, Any]) -> None:
    """
    Validate schema structure.

    Args:
        schemas: Dictionary of schemas to validate

    Raises:
        ValueError: If schema structure is invalid
    """
    if not isinstance(schemas, dict):
        raise ValueError("Schemas must be a dictionary")

    for table_name, schema in schemas.items():
        if not isinstance(schema, dict):
            raise ValueError(f"Schema for table '{table_name}' must be a dictionary")

        # Validate required fields
        if 'fields' not in schema:
            raise ValueError(f"Schema for table '{table_name}' missing 'fields'")

        if not isinstance(schema['fields'], list):
            raise ValueError(f"Fields for table '{table_name}' must be a list")

        # Validate each field
        for field in schema['fields']:
            if not isinstance(field, dict):
                raise ValueError(f"Field in table '{table_name}' must be a dictionary")

            if 'name' not in field or 'type' not in field:
                raise ValueError(f"Field in table '{table_name}' missing 'name' or 'type'")


def get_table_description(
    table_name: str,
    schema: Dict[str, Any],
    include_descriptions: bool = True,
    context_limit: Optional[int] = None,
    embedding_model: Optional[str] = None
) -> str:
    """
    Get a formatted description of a table for embeddings.

    Args:
        table_name: Name of the table
        schema: Schema definition for the table
        include_descriptions: Whether to include field descriptions (default: True)
        context_limit: Optional token limit for the description (default: None = unlimited)
        embedding_model: Embedding model name for tokenization (required if context_limit is set)

    Returns:
        Formatted string describing the table

    Raises:
        ValueError: If context_limit is set but embedding_model is not provided
    """
    description = schema.get('description', '')
    category = schema.get('category', '')
    fields = schema.get('fields', [])

    # Validate parameters
    if context_limit is not None and embedding_model is None:
        raise ValueError("embedding_model must be specified when context_limit is set")

    # Initialize tokenizer if context_limit is specified
    tokenizer = None
    if context_limit is not None:
        tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{embedding_model}")

    # Start building parts
    parts = [
        f"Table: {table_name}",
        f"Category: {category}" if category else None,
        f"Description: {description}" if description else None,
    ]

    # Remove None values
    parts = [p for p in parts if p]

    # Add fields
    if include_descriptions:
        # Include full field name + description
        field_parts = []
        for field in fields:
            field_desc = f"{field['name']}: {field.get('description', field['type'])}"
            field_parts.append(field_desc)
    else:
        # Just field names
        field_parts = [f['name'] for f in fields]

    # If context_limit specified, truncate to fit
    if context_limit is not None and tokenizer is not None:
        base_text = "\n".join(parts)
        base_tokens = len(tokenizer.encode(base_text))

        remaining_tokens = context_limit - base_tokens

        if remaining_tokens > 0:
            # Add fields until we hit the limit
            included_fields = []
            current_text = ""

            for field_part in field_parts:
                if include_descriptions:
                    test_text = current_text + "\n  - " + field_part if current_text else "  - " + field_part
                else:
                    test_text = current_text + ", " + field_part if current_text else field_part

                test_tokens = len(tokenizer.encode(test_text))

                if test_tokens <= remaining_tokens:
                    included_fields.append(field_part)
                    current_text = test_text
                else:
                    break

            if included_fields:
                if include_descriptions:
                    fields_text = "Fields:\n  - " + "\n  - ".join(included_fields)
                else:
                    fields_text = f"Fields: {', '.join(included_fields)}"
                parts.append(fields_text)

            if len(included_fields) < len(field_parts):
                parts.append(f"... and {len(field_parts) - len(included_fields)} more fields")
    else:
        # No limit - include all fields
        if include_descriptions:
            fields_text = "Fields:\n  - " + "\n  - ".join(field_parts)
        else:
            fields_text = f"Fields: {', '.join(field_parts)}"
        parts.append(fields_text)

    return "\n".join(parts)


def format_schema_for_llm(table_name: str, schema: Dict[str, Any], max_fields: int = 50) -> str:
    """
    Format a schema for LLM context.

    Args:
        table_name: Name of the table
        schema: Schema definition
        max_fields: Maximum number of fields to include

    Returns:
        Formatted schema string for LLM
    """
    description = schema.get('description', '')
    category = schema.get('category', '')
    fields = schema.get('fields', [])[:max_fields]

    result = [f"Table: {table_name}"]

    if category:
        result.append(f"Category: {category}")

    if description:
        result.append(f"Description: {description}")

    result.append("\nFields:")
    for field in fields:
        field_desc = f"  - {field['name']} ({field['type']})"
        if 'description' in field:
            field_desc += f": {field['description']}"
        result.append(field_desc)

    if len(schema.get('fields', [])) > max_fields:
        result.append(f"  ... and {len(schema['fields']) - max_fields} more fields")

    return "\n".join(result)
