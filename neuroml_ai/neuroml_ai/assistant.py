#!/usr/bin/env python3
"""
Assistant combining various (RAG/code agents)

File: assistant.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


import logging
import os
import sys
from textwrap import dedent
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.utils import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
    parse_output_with_thought,
    setup_llm,
    split_thought_and_output,
)
from pydantic import create_model
from typing_extensions import Dict, List, Literal, Tuple

from .schemas import (AssistantState, QueryTypeSchema)


class NML_Assistant(object):

    """NeuroML Assistant class"""
    def __init__(
        self,
        vs_config_file: str,
        chat_model: str,
        logging_level: int = logging.DEBUG,
    ):
        self.chat_model = chat_model
        self.model = None

        self.checkpointer = InMemorySaver()

        # number of conversations after which to summarise
        # no need to summarise after each
        # 5 rounds: 10 messages
        self.num_recent_messages = 10

        self.logger = logging.getLogger("NML_AI_Assistant")
        self.logger.setLevel(logging_level)
        self.logger.propagate = False

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(LoggerInfoFilter())
        stdout_handler.setFormatter(logger_formatter_info)
        self.logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging_level)
        stderr_handler.addFilter(LoggerNotInfoFilter())
        stderr_handler.setFormatter(logger_formatter_other)
        self.logger.addHandler(stderr_handler)

    def _route_query_node(self, state: AssistantState) -> str:
        """Route the query depending on LLM's result"""
        self.logger.debug(f"{state =}")
        query_type = state.query_type.query_type

        return query_type

    def _classify_query_node(self, state: AssistantState) -> dict:
        """LLM decides what type the user query is"""
        assert self.model
        self.logger.debug(f"{state =}")

        messages = state.messages
        messages.append(HumanMessage(content=state.query))

        system_prompt = dedent("""
            You are an expert query classifier.
            Classify the user input into exactly one category based on its intent

            Valid categories (in order of priority):

            - question: The query is a request for information about NeuroML.
            - task: The user is asking you to perform an action. The action will be performed in a later step.

            Rules:

            - Choose exactly ONE category
            - Base your decision on semantic intent
            - Do not explain your reasoning
            - Do not include any other additional text
            - Provide your answer ONLY as a JSON object matching the requested schema.
            - Take past conversation history and context into account.

            Examples:

            - "How do I get learn NeuroML?": {{"query_type": "question"}}
            - "How do I get started with NeuroML?": {{"query_type": "question"}}
            - "How do I define ion channels in NeuroML?": {{"query_type": "question"}}
            - "Generate NeuroML code for a neuron": {{"query_type": "task"}}
            - "Run this code": {{"query_type": "task"}}
            - "Run this command": {{"query_type": "task"}}
            - "Run this simulation": {{"query_type": "task"}}
            - "What is the capital of France?": {{"query_type": "question"}}
            - "What are we talking about?": {{"query_type": "question"}}
            """)

        # only if neuroml is not mentioned, do we even bother with non neuroml
        # classifications
        # if "neuroml" not in state.query.lower():
        #     system_prompt += dedent("""
        #         - If the query is unrelated to NeuroML, only then respond "general_question".
        #         """)

        system_prompt += self._add_memory_to_prompt(state)

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        # can use | to merge these lines
        query_node_llm = self.model.with_structured_output(
            QueryTypeSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke({"query": state.query})

        self.logger.debug(f"{prompt = }")

        output = query_node_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )
        if output["parsing_error"]:
            query_type_result = parse_output_with_thought(
                output["raw"], QueryTypeSchema
            )
        else:
            query_type_result = output["parsed"]
            if isinstance(query_type_result, str):
                query_type_result = QueryTypeSchema(query_type=query_type_result)
            elif isinstance(query_type_result, dict):
                query_type_result = QueryTypeSchema(**query_type_result)
            else:
                if not isinstance(query_type_result, QueryTypeSchema):
                    self.logger.critical(
                        f"Received unexpected query classification: {query_type_result =}"
                    )
                    query_type_result = QueryTypeSchema(query_type="undefined")

        self.logger.debug(f"{query_type_result =}")
        return {
            "query_type": query_type_result,
            "messages": messages,
        }

