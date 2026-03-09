#!/usr/bin/env python3
"""
Schema definitions

File: neuroml_ai/schemas.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing import Any
from typing_extensions import Dict, List, Literal, Tuple


class QueryTypeSchema(BaseModel):
    """Schema for query type.

    Values:
    - undefined: fallback, treated as a general question routed to RAG
    - question: user wants information, answered via RAG
    - simulation: user wants to run a neuron simulation (HH or LEMS)
    - database: user wants to query a biological database (WormBase, WormAtlas)
    - task: user wants code generated or a coding task performed
    """

    query_type: Literal[
        "undefined",
        "question",
        "simulation",
        "database",
        "task",
    ] = Field(default="undefined")


class AssistantState(BaseModel):
    """State shared across all nodes in the assistant graph"""

    query: str = ""
    query_type: QueryTypeSchema = QueryTypeSchema()
    messages: List[AnyMessage] = Field(default_factory=list)

    # summarised version of context so far
    context_summary: str = ""

    message_for_user: str = ""

    # reference material from RAG retrievals
    # structure: {domain_name: [(Document, relevance_score), ...]}
    reference_material: Dict[str, List[Tuple]] = Field(default_factory=dict)

    # raw outputs from tool calls (simulation or database)
    # structure: {tool_name: {tool output dict}}
    # e.g. {"run_hh_simulation": {"firing_rate_hz": 32.0, ...}}
    #      {"query_wormbase_gene": {"gene": "eat-4", "overview": ...}}
    tool_outputs: Dict[str, Any] = Field(default_factory=dict)