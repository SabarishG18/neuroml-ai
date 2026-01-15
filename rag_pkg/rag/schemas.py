#!/usr/bin/env python3
"""
Schemas used in the RAG

File: gen_rag/schemas.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastmcp.client.client import CallToolResult
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import Any, Dict, List, Literal, Tuple


class QueryTypeSchema(BaseModel):
    """Schema for query type."""

    query_type: Literal["undefined", "question", "task"] = Field(
        default="undefined",
    )


class QueryDomainSchema(BaseModel):
    """Schema for query type."""

    query_domain: Literal["undefined", "neuroml", "general"] = Field(
        default="undefined",
    )


class EvaluateAnswerSchema(BaseModel):
    """Evaluation of LLM generated answer. Descriptions given in the main prompt"""

    confidence: float = 0.0
    coverage: float = 0.0
    relevance: float = 0.0
    groundedness: float = 0.0
    coherence: float = 0.0
    conciseness: float = 0.0
    next_step: Literal[
        "continue", "retrieve_more_info", "modify_query", "rewrite_answer", "undefined"
    ] = Field(default="undefined")
    summary: str = ""


class EvaluateCodeCommandSchema(BaseModel):
    """Schema for code generation or command call."""
    next_step: Literal[
        "continue", "update_code", "call_tool", "undefined"
    ] = Field(default="undefined")
    reason: str = ""

class ToolCallSchema(BaseModel):
    """Schema for tool call response."""
    tool: str = ""
    args: Dict[str, str] = Field(default_factory=dict)
    reason: str = ""


class CodeSchema(BaseModel):
    code: str = ""
    version: int = 0
    patches: List[str] = []


class AgentState(BaseModel):
    """The state of the graph"""

    query: str = ""
    query_type: QueryTypeSchema = QueryTypeSchema()
    query_domain: QueryDomainSchema = QueryDomainSchema()
    text_response_eval: EvaluateAnswerSchema = EvaluateAnswerSchema()
    messages: List[AnyMessage] = Field(default_factory=list)

    # code string if any
    code: CodeSchema = CodeSchema()
    code_eval: EvaluateCodeCommandSchema = EvaluateCodeCommandSchema()

    # tool call response
    tool_call: ToolCallSchema = ToolCallSchema()
    tool_response: CallToolResult = None

    # summarised version of context so far
    context_summary: str = ""

    # index till which summarised
    summarised_till: int = 0
    message_for_user: str = ""

    # reference material from retrievals
    reference_material: Dict[str, List[Tuple]] = Field(default_factory=dict)
