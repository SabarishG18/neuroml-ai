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
    id_: int = 1
    summary: str = ""
    tool_call: bool = False
    inputs: str = ""
    output: str = ""
    status: Literal["pending", "done", "failed"] = Field(default="pending")


class PlanSchema(BaseModel):
    steps: List[StepSchema] = Field(default_factory=list)
    status: Literal["not_started", "in_progress", "completed", "failed", "aborted"] = (
        Field(default="not_started")
    )
    current_step: int = 0


class GoalSchema(BaseModel):
    goal: str = ""
    success_criteria: str = ""


class ArtefactSchema(BaseModel):
    id_: str = ""
    type_: str = ""
    content: Any
    metadata: dict[str, Any] = {}


class CodeAIState(BaseModel):
    """The state of the graph"""

    query: str = ""
    messages: List[AnyMessage] = Field(default_factory=list)

    # code string if any
    code: CodeSchema = CodeSchema()

    # planning related
    goal: GoalSchema = GoalSchema()
    plan: PlanSchema = PlanSchema()

    # current tool call
    tool_call: Optional[ToolCallSchema] = None
    # keep history of responses for context
    tool_responses: list[CallToolResult] = Field(default_factory=list)

    # { id -> artefact }
    artefacts: Dict[str, ArtefactSchema] = Field(default_factory=dict)

    # summarised version of context so far
    context_summary: str = ""

    # index till which summarised
    summarised_till: int = 0
    message_for_user: str = ""
