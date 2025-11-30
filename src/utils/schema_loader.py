"""Schema loader for security events database."""

import json
from pathlib import Path
from typing import Dict, Any


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


def get_table_description(table_name: str, schema: Dict[str, Any]) -> str:
    """
    Get a formatted description of a table for embeddings.

    Args:
        table_name: Name of the table
        schema: Schema definition for the table

    Returns:
        Formatted string describing the table
    """
    description = schema.get('description', '')
    category = schema.get('category', '')
    fields = schema.get('fields', [])

    # Create a comprehensive description for embedding
    field_names = [f['name'] for f in fields]
    field_descriptions = [f"{f['name']}: {f.get('description', '')}" for f in fields[:10]]  # First 10 fields

    parts = [
        f"Table: {table_name}",
        f"Category: {category}" if category else None,
        f"Description: {description}" if description else None,
        f"Key fields: {', '.join(field_names[:15])}",  # First 15 field names
    ]

    return "\n".join(p for p in parts if p)


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
