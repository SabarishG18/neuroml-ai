#!/usr/bin/env python3
"""
Schema definitions

File: schemas.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""
from pydantic import BaseModel, Field
from typing_extensions import Literal
from gen_rag.schemas import RAGState

class QueryTypeSchema(BaseModel):
    """Schema for query type."""

    query_type: Literal["undefined", "question", "task"] = Field(
        default="undefined",
    )


class AssistantState(BaseModel):
    query_type: QueryTypeSchema

    # For question branch
    QAState: RAGState

    # for code branch
    # CodeState: CodeAgentState
