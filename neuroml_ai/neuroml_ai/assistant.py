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

from gen_rag.rag import RAG
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.llm import (
    add_memory_to_prompt,
    parse_output_with_thought,
    setup_llm,
)
from neuroml_ai_utils.logging import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
)

from .schemas import AssistantState, QueryTypeSchema


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
        self.vs_config_file = vs_config_file
        self.checkpointer = InMemorySaver()
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

    def _init_rag_state_node(self, state: AssistantState) -> dict:
        """Initialise, reset state before next iteration"""
        return {
            "query_type": QueryTypeSchema(),
            "message_for_user": "",
            "reference_material": {},
        }

    def _classify_query_node(self, state: AssistantState) -> dict:
        """LLM decides what type the user query is"""
        assert self.model
        self.logger.debug(f"{state =}")

        messages = state.messages
        messages.append(HumanMessage(content=state.query))

        system_prompt = dedent("""
            You are an expert query classifier.
            Classify the user input into exactly one category based on its intent.

            Valid categories:

            - question: The query is a request for information.
            - task: The user is asking you to perform a coding task such as
              generating or running code.

            Rules:

            - Choose exactly ONE category
            - Base your decision on semantic intent
            - Do not explain your reasoning
            - Do not include any other additional text
            - Provide your answer ONLY as a JSON object matching the requested schema.
            - Take past conversation history and context into account.

            Examples:

            - "How do I get started with NeuroML?": {{"query_type": "question"}}
            - "How do I define ion channels in NeuroML?": {{"query_type": "question"}}
            - "How many neurons does C. elegans have?": {{"query_type": "question"}}
            - "What does the AWC neuron do?": {{"query_type": "question"}}
            - "What is the capital of France?": {{"query_type": "question"}}
            - "Generate NeuroML code for a neuron": {{"query_type": "task"}}
            - "Run this code": {{"query_type": "task"}}
            - "Run this command": {{"query_type": "task"}}
            - "Run this simulation": {{"query_type": "task"}}
            """)

        system_prompt += add_memory_to_prompt(
            messages=state.messages,
            context_summary=state.context_summary,
            num_recent_messages=self.num_recent_messages,
        )

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

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

    def _route_query_node(self, state: AssistantState) -> str:
        """Route the query depending on LLM's result"""
        self.logger.debug(f"{state =}")
        return state.query_type.query_type

    def _route_after_rag(self, state: AssistantState) -> str:
        """Route after RAG completes.

        For task queries, RAG has already populated reference_material.
        Route to code_graph so it can use that context.
        For everything else, RAG has already generated the answer.
        """
        if state.query_type.query_type == "task":
            return "code_graph"
        return END

    def _code_node(self, state: AssistantState):
        """Code node with RAG context available.

        state.reference_material contains literature retrieved by the RAG
        subgraph, which can be used to inform code generation (e.g. knowing
        AWC neuron properties when asked to generate a model for it).

        TODO: implement RAG-informed code generation using MCP tools.
        """
        return {"message_for_user": "Coding support is a work in progress"}

    def _setup_chat_model(self):
        """Set up the LLM chat model"""
        self.model = setup_llm(self.chat_model, self.logger)

    async def setup(self):
        """Set up basics."""
        self._setup_chat_model()
        await self._create_graph()

    async def _create_graph(self):
        """Create the LangGraph.

        Graph structure:

            classify → question/undefined → rag_graph → END
                     → task              → rag_graph → code_graph → END

        Questions go through RAG which generates the answer directly.
        Tasks also go through RAG first (to populate reference_material
        with relevant literature context), then to the code node which
        can use that context for better code generation.

        Simulation, database, and other tool integrations hook in via
        MCP servers (NeuroML MCP, OpenWorm MCP), not as graph nodes.
        """
        self.workflow = StateGraph(AssistantState)

        self.workflow.add_node("init_state", self._init_rag_state_node)
        self.workflow.add_node("classify_query", self._classify_query_node)
        self.workflow.add_node("code_graph", self._code_node)

        # RAG subgraph
        self._rag_node = RAG(
            vs_config_file=self.vs_config_file,
            chat_model=self.chat_model,
            memory=False
        )
        self._rag_node_graph = await self._rag_node.get_graph()
        self.workflow.add_node("rag_graph", self._rag_node_graph)

        # edges
        self.workflow.add_edge(START, "init_state")
        self.workflow.add_edge("init_state", "classify_query")

        # classify routes: questions go to RAG, tasks also go to RAG
        # (everything goes through RAG first)
        self.workflow.add_conditional_edges(
            "classify_query",
            self._route_query_node,
            {
                "undefined": "rag_graph",
                "question":  "rag_graph",
                "task":      "rag_graph",
            },
        )

        # after RAG: questions are done (RAG answered), tasks go to code
        self.workflow.add_conditional_edges(
            "rag_graph",
            self._route_after_rag,
            {
                END:          END,
                "code_graph": "code_graph",
            },
        )

        self.workflow.add_edge("code_graph", END)

        self.graph = self.workflow.compile(checkpointer=self.checkpointer)

        if not os.environ.get("RUNNING_IN_DOCKER", 0):
            try:
                self.graph.get_graph().draw_mermaid_png(
                    output_file_path="nml-ai-assistant-lang-graph.png"
                )
            except BaseException as e:
                self.logger.error("Something went wrong generating lang graph png")
                self.logger.error(e)

    async def run_graph_invoke_state(
        self, state: dict, thread_id: str = "default_thread"
    ):
        """Run the graph but accept and return states"""
        config = {"configurable": {"thread_id": thread_id}}
        if "query" not in state:
            self.logger.error(f"Provided state should include the key 'query': {state}")
            sys.exit(-1)
        final_state = await self.graph.ainvoke(state, config=config)
        self.logger.debug(final_state)
        return final_state

    async def run_graph_invoke(self, query: str, thread_id: str = "default_thread"):
        """Run the graph by using and returning string input"""
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await self.graph.ainvoke({"query": query}, config=config)
        self.logger.debug(f"{final_state =}")
        if message := final_state.get("message_for_user", None):
            return message
        else:
            return "I was unable to answer"

    async def run_graph_stream(self, query: str, thread_id: str = "default_thread"):
        """Run the graph but return the stream"""
        config = {"configurable": {"thread_id": thread_id}}
        for chunk in self.graph.astream({"query": query}, config=config):
            for node, state in chunk.items():
                self.logger.debug(f"{node}: {repr(state)}")
                if message := state.get("message_for_user", None):
                    self.logger.info(f"User message: {message}")
                    yield message
                else:
                    self.logger.debug(f"Working in node: {node}")

    async def graph_stream(self, query: str, thread_id: str = "default_thread"):
        """Run the graph but return the stream"""
        config = {"configurable": {"thread_id": thread_id}}
        res = await self.graph.astream({"query": query}, config=config)
        return res
