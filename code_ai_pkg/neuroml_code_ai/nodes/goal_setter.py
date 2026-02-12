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
from neuroml_code_ai.schemas import GoalSchema


class GoalSetterNode(BaseCodeAINode):
    """Goal setter node"""

    def __init__(
        self, logger, model, system_prompt_template, human_prompt_template, memory
    ):
        BaseCodeAINode.__init__(self, logger, model)

    def _get_output_schema(self):
        """Return Pydantic schema for structured output"""
        return GoalSchema

    def _create_prompt_template(
        self, system_prompt: str, human_prompt: str
    ) -> ChatPromptTemplate:
        """Create ChatPromptTemplate for this node"""
        pass

    def _get_prompt_variables(self, state: BaseModel) -> dict:
        """Format prompt with state-specific parameters"""
        return {}

    def _update_state(self, result: BaseModel, state: BaseModel) -> Dict[str, Any]:
        """Update and return state dictionary"""
        return {}

    def _get_default_error_result(self) -> BaseModel:
        """Return default result when processing fails"""
        pass
