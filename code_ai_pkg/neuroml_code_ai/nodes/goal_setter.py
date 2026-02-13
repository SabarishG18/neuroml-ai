#!/usr/bin/env python3
"""
Goal setter node

File: code_ai_pkg/neuroml_code_ai/nodes/goal_setter.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from neuroml_code_ai.nodes.base_node import BaseCodeAINode
from neuroml_code_ai.schemas import CodeAIState, GoalSchema


class GoalSetterNode(BaseCodeAINode[GoalSchema]):
    """Goal setter node"""

    def __init__(
        self,
        logger,
        model,
        temperature,
        output_schema,
        system_prompt_file,
        human_prompt_file,
        memory,
    ):
        super().__init__(
            self,
            logger,
            model,
            temperature,
            output_schema,
            system_prompt_file,
            human_prompt_file,
            memory,
        )

    def _create_prompt_template(
        self, system_prompt: str, human_prompt: str
    ) -> ChatPromptTemplate:
        """Create ChatPromptTemplate for this node"""
        pass

    def _get_prompt_variables(self, state: CodeAIState) -> dict:
        """Format prompt with state-specific parameters"""
        return {"query": state.query, "context_summary": ""}

    def _update_state(self, result: GoalSchema, state: BaseModel) -> Dict[str, Any]:
        """Update and return state dictionary"""
        return {"goal": result, "message_for_user": result.goal}

    def _get_default_error_result(self) -> GoalSchema:
        """Return default result when processing fails"""
        return self._get_output_schema()(goal="Invalid", success_criteria="Invalid")
