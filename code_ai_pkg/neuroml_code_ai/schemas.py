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
from typing_extensions import Any, Dict, List, Literal, Optional, Tuple


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


class ToolCallSchema(BaseModel):
    """Schema for tool call response."""

    tool: str = ""
    args: Dict[str, str] = Field(default_factory=dict)
    reason: str = ""


class CodeSchema(BaseModel):
    code: str = ""
    version: int = 0
    patches: List[str] = []


class StepSchema(BaseModel):
    id_: int = 0
    step_summary: str = ""
    tool_call: Optional[ToolCallSchema]
    inputs: Dict[str, str] = Field(default_factory=dict)
    produces: Dict[str, str] = Field(default_factory=dict)
    status: Literal["pending", "done", "failed"] = Field(default="pending")


class PlanSchema(BaseModel):
    plan: List[StepSchema] = Field(default_factory=list)
    plan_status: Literal[
        "not_started", "in_progress", "completed", "failed", "aborted"
    ] = Field(default="not_started")
    current_plan_step: int = 0


class GoalSchema(BaseModel):
    goal: str = ""
    success_criteria: str = ""


class ArtefactSchema(BaseModel):
    id_: str = ""
    type_: str = ""
    content: Any
    metadata: dict[str, Any] = {}


class AgentState(BaseModel):
    """The state of the graph"""

    query: str = ""
    query_domain: QueryDomainSchema = QueryDomainSchema()
    messages: List[AnyMessage] = Field(default_factory=list)

    # code string if any
    code: CodeSchema = CodeSchema()

    # planning related
    goal: GoalSchema = GoalSchema()
    plan: PlanSchema = PlanSchema()

    tool_responses: Dict[int, CallToolResult] = {}

    iteration_count: int = 0
    max_iterations: int = 20
    # { id -> artefact }
    artefacts: Dict[str, ArtefactSchema] = Field(default_factory=dict)

    # summarised version of context so far
    context_summary: str = ""

    # index till which summarised
    summarised_till: int = 0
    message_for_user: str = ""
