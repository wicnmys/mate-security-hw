"""
Shared agent registry for CLI and experiments.

This module provides a centralized registry of all available agents,
making it easy to add new agents in one place.
"""

from typing import Dict, Type, Any, Optional

from src.agents.base import BaseAgent
from src.agents.keyword_agent import KeywordAgent
from src.agents.semantic_agent import SemanticAgent
from src.agents.react_agent import ReActAgent
from src.agents.react_agent_v2 import ReActAgentV2


# Central registry mapping agent names to their classes
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "keyword": KeywordAgent,
    "semantic": SemanticAgent,
    "react": ReActAgent,
    "react-v2": ReActAgentV2,
}

# Default configurations for each agent type (used by experiments)
AGENT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "keyword": {
        "retrieval_type": "keyword",
    },
    "semantic": {
        "retrieval_type": "semantic",
    },
    "react": {
        "retrieval_type": "semantic",
    },
    "react-v2": {
        "retrieval_type": "semantic",
    },
}


def get_agent_names() -> list[str]:
    """Return list of available agent names."""
    return list(AGENT_REGISTRY.keys())


def get_agent_class(name: str) -> Type[BaseAgent]:
    """
    Get the agent class for a given name.

    Args:
        name: Agent name (e.g., 'semantic', 'react-v2')

    Returns:
        The agent class

    Raises:
        KeyError: If agent name is not found
    """
    if name not in AGENT_REGISTRY:
        available = ", ".join(AGENT_REGISTRY.keys())
        raise KeyError(f"Unknown agent '{name}'. Available: {available}")
    return AGENT_REGISTRY[name]


def create_agent(
    name: str,
    schema_path: str,
    top_k_tables: int = 5,
    model: Optional[str] = None,
    **kwargs
) -> BaseAgent:
    """
    Factory function to create an agent instance.

    Args:
        name: Agent name (e.g., 'semantic', 'react-v2')
        schema_path: Path to the schema JSON file
        top_k_tables: Number of tables to retrieve
        model: LLM model to use (optional)
        **kwargs: Additional agent-specific arguments

    Returns:
        An instantiated agent
    """
    agent_class = get_agent_class(name)

    # Merge defaults with provided kwargs
    defaults = AGENT_DEFAULTS.get(name, {})
    merged_kwargs = {**defaults, **kwargs}

    # Build constructor args
    init_args = {
        "schema_path": schema_path,
        "top_k_tables": top_k_tables,
    }

    if model:
        init_args["model"] = model

    # Add retrieval_type for agents that support it
    if "retrieval_type" in merged_kwargs and name in ("react", "react-v2"):
        init_args["retrieval_type"] = merged_kwargs["retrieval_type"]

    # Add judge_model for react-v2
    if name == "react-v2" and "judge_model" in merged_kwargs:
        init_args["judge_model"] = merged_kwargs["judge_model"]

    return agent_class(**init_args)


def get_agent_config(
    name: str,
    schema_path: str,
    top_k: int,
    llm_model: str,
    embedding_model: str,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate experiment config metadata for an agent.

    Args:
        name: Agent name
        schema_path: Path to schema file
        top_k: Number of tables to retrieve
        llm_model: LLM model name
        embedding_model: Embedding model name
        judge_model: Judge model (for react-v2)

    Returns:
        Config dict for experiment tracking
    """
    defaults = AGENT_DEFAULTS.get(name, {})
    retrieval_type = defaults.get("retrieval_type", "unknown")

    config = {
        "type": name,
        "llm_model": llm_model,
        "top_k": top_k,
        "schema_path": schema_path,
    }

    # Add embedding model for semantic-based agents
    if retrieval_type == "semantic":
        config["embedding_model"] = embedding_model
        config["retrieval_type"] = "semantic"

    # Add judge model for react-v2
    if name == "react-v2" and judge_model:
        config["judge_model"] = judge_model

    return config
