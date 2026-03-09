#!/usr/bin/env python3
"""
Assistant combining various (RAG/code agents)

File: assistant.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
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

# CHANGE 1: Import tool functions directly for use in tool nodes
from neuroml_mcp.tools.hh_tools import run_hh_simulation_tool
from neuroml_mcp.tools.neuroml_tools import run_lems_simulation
from neuroml_mcp.tools.wormbase_tools import (
    query_wormbase_gene_tool,
    query_wormbase_neuron_tool,
    search_wormbase_tool,
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
        # UNCHANGED from original
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
        # UNCHANGED from original
        return {
            "query_type": QueryTypeSchema(),
            "message_for_user": "",
            "reference_material": {},
            # CHANGE 2: also reset tool_outputs on each new query
            "tool_outputs": {},
        }

    def _classify_query_node(self, state: AssistantState) -> dict:
        """LLM decides what type the user query is"""
        # UNCHANGED except for the system prompt examples which are extended
        assert self.model
        self.logger.debug(f"{state =}")

        messages = state.messages
        messages.append(HumanMessage(content=state.query))

        # CHANGE 3: Extended system prompt to cover C. elegans, OpenWorm,
        # and the two new query types: "simulation" and "database"
        # Original only had "question" and "task" with NeuroML-only examples
        system_prompt = dedent("""
            You are an expert query classifier.
            Classify the user input into exactly one category based on its intent.

            Valid categories (in order of priority):

            - question: The query is a request for information about NeuroML,
              C. elegans biology, neuroscience, or the OpenWorm project.
            - simulation: The user wants to run a neuron simulation. This includes
              Hodgkin-Huxley simulations, LEMS simulations, or NeuroML model simulations.
            - database: The user wants to look up biological data from a database
              such as WormBase or WormAtlas. This includes gene lookups, neuron
              anatomy queries, phenotype queries, and expression data.
            - task: The user is asking you to perform a coding task such as
              generating or running code. The action will be performed in a later step.

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
            - "What neurotransmitter does eat-4 encode?": {{"query_type": "question"}}
            - "Explain the connectome of C. elegans": {{"query_type": "question"}}
            - "What is the capital of France?": {{"query_type": "question"}}
            - "Run a Hodgkin-Huxley simulation with 0.2nA current injection": {{"query_type": "simulation"}}
            - "What happens to firing rate if I double the current?": {{"query_type": "simulation"}}
            - "Simulate a neuron for 500ms": {{"query_type": "simulation"}}
            - "Run a LEMS simulation of this model": {{"query_type": "simulation"}}
            - "Look up eat-4 on WormBase": {{"query_type": "database"}}
            - "What phenotype does unc-17 cause?": {{"query_type": "database"}}
            - "What genes are expressed in AWC?": {{"query_type": "database"}}
            - "Find the WormBase entry for the AIY neuron": {{"query_type": "database"}}
            - "Generate NeuroML code for a neuron": {{"query_type": "task"}}
            - "Run this code": {{"query_type": "task"}}
            - "Run this command": {{"query_type": "task"}}
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
        # UNCHANGED - just returns the query_type string which LangGraph
        # uses to pick the next node via conditional edges
        self.logger.debug(f"{state =}")
        return state.query_type.query_type

    # CHANGE 4: New simulation node
    # Replaces the old _code_node for simulation queries.
    # Internally decides between HH and LEMS based on the query.
    # Designed to be extended with more simulation types in future.
    async def _simulation_node(self, state: AssistantState) -> dict:
        """Determine simulation type from query and run appropriate tool.

        Currently supports:
        - Hodgkin-Huxley (HH) via NEURON: for general neuron firing queries
        - LEMS via pynml: for running specific NeuroML model files

        Future extensions:
        - OpenWorm muscle models
        - Full connectome simulations
        """
        assert self.model
        self.logger.debug(f"{state =}")

        # First, use LLM to extract simulation parameters from natural language
        param_extraction_prompt = dedent("""
            You are a neuroscience simulation parameter extractor.

            Given a user query about a neuron simulation, extract the relevant
            parameters and simulation type.

            Respond ONLY as a JSON object with these keys:

            {{
                "simulation_type": "hh" or "lems",
                "current_injection": float or null,
                "duration": float or null,
                "delay": float or null,
                "temperature": float or null,
                "lems_file": string or null
            }}

            Rules:
            - Use "hh" for Hodgkin-Huxley or general neuron firing queries
            - Use "lems" only if a specific LEMS file or NeuroML model file is mentioned
            - Use null for any parameter not mentioned in the query
            - current_injection is in nA, duration and delay are in ms, temperature in Celsius

            Examples:
            "Run a simulation with 0.2nA current" →
                {{"simulation_type": "hh", "current_injection": 0.2, "duration": null, "delay": null, "temperature": null, "lems_file": null}}
            "What happens with 0.5nA for 500ms?" →
                {{"simulation_type": "hh", "current_injection": 0.5, "duration": 500.0, "delay": null, "temperature": null, "lems_file": null}}
            "Run LEMS_sim.xml" →
                {{"simulation_type": "lems", "current_injection": null, "duration": null, "delay": null, "temperature": null, "lems_file": "LEMS_sim.xml"}}
            """)

        prompt_template = ChatPromptTemplate(
            [("system", param_extraction_prompt), ("human", "User query: {query}")]
        )
        prompt = prompt_template.invoke({"query": state.query})
        output = self.model.invoke(
            prompt, config={"configurable": {"temperature": 0.0}}
        )

        # parse extracted parameters
        try:
            raw = output.content.strip()
            # strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            params = json.loads(raw.strip())
        except Exception as e:
            self.logger.error(f"Failed to parse simulation parameters: {e}")
            # fall back to defaults
            params = {"simulation_type": "hh"}

        sim_type = params.get("simulation_type", "hh")
        self.logger.debug(f"Simulation type: {sim_type}, params: {params}")

        tool_result = {}

        if sim_type == "lems":
            lems_file = params.get("lems_file")
            if lems_file:
                self.logger.debug(f"Running LEMS simulation: {lems_file}")
                tool_result = await run_lems_simulation(lems_file=lems_file)
            else:
                tool_result = {"error": "No LEMS file specified in query"}

        else:
            # default to HH for all other simulation queries
            # build kwargs, only passing params that were explicitly extracted
            hh_kwargs = {}
            if params.get("current_injection") is not None:
                hh_kwargs["current_injection"] = float(params["current_injection"])
            if params.get("duration") is not None:
                hh_kwargs["duration"] = float(params["duration"])
            if params.get("delay") is not None:
                hh_kwargs["delay"] = float(params["delay"])
            if params.get("temperature") is not None:
                hh_kwargs["temperature"] = float(params["temperature"])

            self.logger.debug(f"Running HH simulation with kwargs: {hh_kwargs}")
            raw_result = await run_hh_simulation_tool(**hh_kwargs)

            # parse the JSON from stdout
            try:
                tool_result = json.loads(raw_result["stdout"])
            except Exception as e:
                self.logger.error(f"Failed to parse HH simulation output: {e}")
                tool_result = raw_result

        return {
            "tool_outputs": {"simulation": tool_result}
        }

    # CHANGE 5: New database node
    # Handles WormBase queries by extracting what to look up from the query
    # and calling the appropriate wormbase tool function.
    async def _database_node(self, state: AssistantState) -> dict:
        """Determine what to query from WormBase and run the appropriate tool.

        Handles:
        - Gene lookups (overview, phenotype, expression, interactions)
        - Neuron/anatomy term lookups
        - General searches when identifier is unclear
        """
        assert self.model
        self.logger.debug(f"{state =}")

        # use LLM to extract query parameters from natural language
        param_extraction_prompt = dedent("""
            You are a C. elegans database query parameter extractor.

            Given a user query about C. elegans biology, extract what to look
            up in WormBase and respond ONLY as a JSON object:

            {{
                "entity_type": "gene" or "neuron" or "search",
                "identifier": string,
                "field": string or null
            }}

            Rules:
            - Use "gene" if the query mentions a gene name (e.g. eat-4, unc-17, glr-1)
            - Use "neuron" if the query mentions a specific neuron (e.g. AWC, AIY, AVAL)
            - Use "search" if unclear or only a keyword is given
            - identifier is the gene name, neuron name, or search keyword
            - field is one of: overview, phenotype, expression, function,
              interactions, references — pick the most relevant one, or null for overview

            Examples:
            "Look up eat-4 on WormBase" →
                {{"entity_type": "gene", "identifier": "eat-4", "field": "overview"}}
            "What phenotype does unc-17 cause?" →
                {{"entity_type": "gene", "identifier": "unc-17", "field": "phenotype"}}
            "What genes are expressed in AWC?" →
                {{"entity_type": "neuron", "identifier": "AWC", "field": "expressed_in"}}
            "Find information about glutamate signalling" →
                {{"entity_type": "search", "identifier": "glutamate", "field": null}}
            """)

        prompt_template = ChatPromptTemplate(
            [("system", param_extraction_prompt), ("human", "User query: {query}")]
        )
        prompt = prompt_template.invoke({"query": state.query})
        output = self.model.invoke(
            prompt, config={"configurable": {"temperature": 0.0}}
        )

        # parse extracted parameters
        try:
            raw = output.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            params = json.loads(raw.strip())
        except Exception as e:
            self.logger.error(f"Failed to parse database query parameters: {e}")
            params = {"entity_type": "search", "identifier": state.query, "field": None}

        entity_type = params.get("entity_type", "search")
        identifier = params.get("identifier", "")
        field = params.get("field") or "overview"

        self.logger.debug(f"WormBase query: type={entity_type}, id={identifier}, field={field}")

        tool_result = {}

        if entity_type == "gene":
            tool_result = await query_wormbase_gene_tool(
                gene_id=identifier, field=field
            )
        elif entity_type == "neuron":
            tool_result = await query_wormbase_neuron_tool(
                neuron_name=identifier, field=field
            )
        else:
            # search fallback
            tool_result = await search_wormbase_tool(query=identifier)

        return {
            "tool_outputs": {"wormbase": tool_result}
        }

    # CHANGE 6: New synthesis node
    # This is the core new node that combines RAG reference material with
    # tool outputs into a single grounded answer.
    # For pure "question" queries, tool_outputs will be empty and this
    # just uses RAG context — consistent behaviour across all query types.
    def _synthesis_node(self, state: AssistantState) -> dict:
        """Synthesise a final answer from RAG context and tool outputs.

        For question queries: uses RAG reference material only.
        For simulation queries: combines literature context with simulation data.
        For database queries: combines literature context with live database results.

        This is the node that produces the final message_for_user for all
        non-task query types.
        """
        assert self.model
        self.logger.debug(f"{state =}")

        # serialise reference material into readable text
        # reusing the same approach as gen_rag for consistency
        ref_text = ""
        for domain, refs in state.reference_material.items():
            ref_text += f"\n## Retrieved literature ({domain}):\n"
            for i, (doc, score) in enumerate(refs):
                ref_text += f"\n### Source {i+1} (relevance: {score:.2f}):\n"
                ref_text += doc.page_content + "\n"

        # serialise tool outputs
        tool_text = ""
        if state.tool_outputs:
            tool_text = "\n## Empirical data from tools:\n"
            for tool_name, result in state.tool_outputs.items():
                tool_text += f"\n### {tool_name}:\n"
                tool_text += json.dumps(result, indent=2) + "\n"

        # build synthesis prompt depending on what data is available
        if state.tool_outputs:
            # tool-augmented synthesis
            system_prompt = dedent("""
                You are a neuroscience research assistant.
                Answer the user's question by synthesising two sources of evidence:

                1. Retrieved literature context from the knowledge base
                2. Empirical data returned by a tool (simulation or database query)

                # Core Directives:
                - Ground your answer in both the literature context AND the tool data
                - Clearly distinguish what comes from literature vs what comes from the tool
                - Use formal, academic neuroscience language
                - Do not mention "context", "retrieval", "RAG", or "tool" directly
                - Do not include your thinking in your response
                - If the tool returned an error, say so clearly and rely on literature only

                {ref_text}

                {tool_text}
                """)
        else:
            # RAG-only synthesis (for pure question queries)
            system_prompt = dedent("""
                You are a neuroscience research assistant.
                Answer the user's question using the provided literature context.

                # Core Directives:
                - Limit yourself to facts from the provided context only
                - Use formal, academic neuroscience language
                - Do not mention "context", "retrieval", or "documents" directly
                - Write a self-contained answer that does not assume access to the context
                - Do not include your thinking in your response

                {ref_text}
                """)

        system_prompt = system_prompt.format(
            ref_text=ref_text,
            tool_text=tool_text,
        )

        system_prompt += add_memory_to_prompt(
            messages=state.messages,
            context_summary=state.context_summary,
            num_recent_messages=self.num_recent_messages,
        )

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "Question: {query}")]
        )
        prompt = prompt_template.invoke({"query": state.query})
        output = self.model.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )

        messages = state.messages
        messages.append(output)

        return {
            "messages": messages,
            "message_for_user": output.content,
        }

    def _code_node(self, state: AssistantState):
        """Dummy code node - unchanged from original"""
        return {"message_for_user": "Coding support is a work in progress"}

    def _setup_chat_model(self):
        """Set up the LLM chat model - unchanged"""
        self.model = setup_llm(self.chat_model, self.logger)

    async def setup(self):
        """Set up basics - unchanged"""
        self._setup_chat_model()
        await self._create_graph()

    async def _create_graph(self):
        """Create the LangGraph.

        CHANGE 7: New graph structure.

        Old structure:
            classify → question/undefined → rag_graph
                       task              → code_graph

        New structure:
            classify → rag_graph (always, for all non-task queries)
                     → then route by query_type:
                           question/undefined → synthesis_node
                           simulation        → simulation_node → synthesis_node
                           database          → database_node → synthesis_node
                           task              → code_graph (unchanged)

        RAG always runs to provide literature context. Tool nodes run
        additionally for simulation and database queries, adding empirical
        data. Synthesis combines both into the final answer.
        """
        self.workflow = StateGraph(AssistantState)

        # existing nodes - unchanged
        self.workflow.add_node("init_state", self._init_rag_state_node)
        self.workflow.add_node("classify_query", self._classify_query_node)
        self.workflow.add_node("code_graph", self._code_node)

        # RAG subgraph - unchanged, still runs as before
        self._rag_node = RAG(
            vs_config_file=self.vs_config_file,
            chat_model=self.chat_model,
            memory=False
        )
        self._rag_node_graph = await self._rag_node.get_graph()
        self.workflow.add_node("rag_graph", self._rag_node_graph)

        # NEW nodes
        self.workflow.add_node("simulation_node", self._simulation_node)
        self.workflow.add_node("database_node", self._database_node)
        self.workflow.add_node("synthesis_node", self._synthesis_node)

        # edges - start is unchanged
        self.workflow.add_edge(START, "init_state")
        self.workflow.add_edge("init_state", "classify_query")

        # CHANGE: classify now always goes to RAG first (except task)
        self.workflow.add_conditional_edges(
            "classify_query",
            self._route_query_node,
            {
                "undefined":   "rag_graph",
                "question":    "rag_graph",
                "simulation":  "rag_graph",
                "database":    "rag_graph",
                "task":        "code_graph",
            },
        )

        # CHANGE: after RAG, route to tool node or straight to synthesis
        self.workflow.add_conditional_edges(
            "rag_graph",
            self._route_after_rag,
            {
                "simulation":  "simulation_node",
                "database":    "database_node",
                "default":     "synthesis_node",
            },
        )

        # tool nodes feed into synthesis
        self.workflow.add_edge("simulation_node", "synthesis_node")
        self.workflow.add_edge("database_node", "synthesis_node")

        # synthesis and code both go to END
        self.workflow.add_edge("synthesis_node", END)
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

    def _route_after_rag(self, state: AssistantState) -> str:
        """Route after RAG based on original query type.

        CHANGE 8: New routing function called after RAG completes.
        RAG always runs first. This then decides whether to additionally
        run a tool node or go straight to synthesis.

        - simulation/database: run the appropriate tool node first
        - everything else: go straight to synthesis
        """
        query_type = state.query_type.query_type
        if query_type == "simulation":
            return "simulation"
        elif query_type == "database":
            return "database"
        else:
            return "default"

    # run methods - all unchanged from original
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