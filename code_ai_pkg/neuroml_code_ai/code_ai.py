#!/usr/bin/env python3
"""
NeuroML CodeAI implementation

File: code_ai.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.llm import load_prompt, parse_output_with_thought, setup_llm
from neuroml_ai_utils.logging import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
)

from neuroml_code_ai import prompts

from .schemas import CodeAIState, GoalSchema, PlanSchema, ToolCallSchema

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class CodeAI(object):
    """NeuroML CodeAI implementation"""

    checkpointer = InMemorySaver()

    def __init__(
        self,
        mcp_client: Client = None,
        code_model: Optional[str] = None,
        logging_level: int = logging.DEBUG,
        memory: bool = True,
    ):
        """Initialise"""
        self.code_model = "ollama:qwen3:1.7b" if code_model is None else code_model
        self.model = None

        # number of conversations after which to summarise
        # no need to summarise after each
        # 5 rounds: 10 messages
        self.num_recent_messages = 10

        self.mcp_client: Client = mcp_client
        self.mcp_tools = None
        self.tool_description = ""

        self.logger = logging.getLogger("NeuroML-AI-codegen")
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

    async def setup(self):
        """Set up basics."""

        self._setup_code_model()

        if self.mcp_client:
            async with self.mcp_client:
                self.mcp_tools = await self.mcp_client.list_tools()
            # Persists because it's only holding the return value
            # To make the call, though, we will need the context again
            self.logger.debug(f"{self.mcp_tools =}")
        else:
            self.logger.warning(
                "No MCP client available. Functions requiring MCP server will fail."
            )

        # Generate tool description
        # Note: we remove the param/type information from the description and
        # use the type hints directly instead
        ctr = 0
        for t in self.mcp_tools:
            if "dummy" in t.name:
                continue
            ctr += 1
            self.tool_description += f"## {ctr}.  {t.name}\n{t.description}\n"
            # args = t.inputSchema.get("properties", [])
            # if len(args):
            #     self.tool_description += "\n\nInputs:\n"
            #     for arg, arginfo in args.items():
            #         self.tool_description += f"- {arg}: {arginfo.get('type')}"
            #     self.tool_description += "\n"

        self.logger.debug(f"{self.tool_description =}")
        await self._create_graph()

    def _setup_code_model(self):
        """Set up the LLM chat model"""
        self.model = setup_llm(self.code_model, self.logger)

    def _init_graph_state_node(self, state: CodeAIState) -> dict:
        """Initialise, reset state before next iteration"""
        return {
            "message_for_user": "",
            "plan": PlanSchema(),
            "goal": GoalSchema(),
            "tool_call": None,
            "tool_responses": [],
        }

    async def _goal_setter_node(self, state: CodeAIState) -> dict:
        assert self.model
        self.logger.debug(f"{state =}")
        system_prompt = load_prompt(
            prompt_name="goal",
            prompt_registry_location=Path(prompts.__file__).parent,
        )
        self.logger.debug(f"{system_prompt = }")

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )
        OutputSchema = GoalSchema

        # can use | to merge these lines
        goal_setter_llm = self.model.with_structured_output(
            OutputSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke(
            {
                "query": state.query,
                "context_summary": "",  # TODO add memory module
            }
        )

        self.logger.debug(f"{prompt = }")

        output = goal_setter_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.01}}
        )

        self.logger.debug(f"{output = }")

        if output["parsing_error"]:
            goal_result = parse_output_with_thought(output["raw"], OutputSchema)
        else:
            goal_result = output["parsed"]
            if isinstance(goal_result, dict):
                goal_result = OutputSchema(**goal_result)
            else:
                if not isinstance(goal_result, OutputSchema):
                    self.logger.critical(
                        f"Received unexpected query classification: {goal_result =}"
                    )
                    goal_result = OutputSchema(
                        goal="Invalid", success_criteria="Invalid"
                    )

        self.logger.debug(f"{goal_result =}")

        return {"goal": goal_result, "message_for_user": goal_result.goal}

    # TODO: add note that does the goal before planning begins
    async def _planner_node(self, state: CodeAIState) -> dict:
        assert self.model
        self.logger.debug(f"{state =}")

        system_prompt = load_prompt(
            prompt_name="planner",
            prompt_registry_location=Path(prompts.__file__).parent,
        )
        self.logger.debug(f"{system_prompt = }")

        OutputSchema = PlanSchema

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        # can use | to merge these lines
        planner_llm = self.model.with_structured_output(
            OutputSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke(
            {
                "query": state.query,
                "goal": state.goal,
                "plan": state.plan,
                "current_step": state.plan.current_step,
                "artefacts": state.artefacts,
                "observations": state.tool_responses,
                "tools_description": self.tool_description,
            }
        )

        self.logger.debug(f"{prompt = }")

        output = planner_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.01}}
        )

        self.logger.debug(f"{output = }")

        if output["parsing_error"]:
            plan_result = parse_output_with_thought(output["raw"], OutputSchema)
        else:
            plan_result = output["parsed"]
            if isinstance(plan_result, dict):
                plan_result = OutputSchema(**plan_result)
            else:
                if not isinstance(plan_result, OutputSchema):
                    self.logger.critical(
                        f"Received unexpected query classification: {plan_result =}"
                    )
                    plan_result = OutputSchema(status="failed")

        self.logger.debug(f"{plan_result =}")

        # Generate a plan summary and send to user
        plan_summary = "## Plan summary:\n\n"
        for step in plan_result:
            plan_summary += f"- {step.id_}: {step.summary}"

        plan = state.plan
        plan.steps = plan_result

        return {"plan": plan, "message_for_user": plan_summary}

    async def _tool_picker_node(self, state: CodeAIState) -> dict:
        assert self.model
        self.logger.debug(f"{state =}")

        current_step = state.plan.steps[state.plan.current_step]
        if not current_step.tool_required:
            return {}

        system_prompt = load_prompt(
            prompt_name="tool_picker",
            prompt_registry_location=Path(prompts.__file__).parent,
        )
        self.logger.debug(f"{system_prompt = }")

        OutputSchema = ToolCallSchema

        prompt_template = ChatPromptTemplate([("system", system_prompt)])

        # can use | to merge these lines
        planner_llm = self.model.with_structured_output(
            OutputSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke(
            {
                "goal": state.goal,
                "current_step": state.plan.current_step,
                "artefacts": state.artefacts,
                "observations": state.tool_responses,
                "tools_description": self.tool_description,
            }
        )

        self.logger.debug(f"{prompt = }")

        output = planner_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.01}}
        )

        self.logger.debug(f"{output = }")

        if output["parsing_error"]:
            tool_picker_result = parse_output_with_thought(output["raw"], OutputSchema)
        else:
            tool_picker_result = output["parsed"]
            if isinstance(tool_picker_result, dict):
                tool_picker_result = OutputSchema(**tool_picker_result)
            else:
                if not isinstance(tool_picker_result, OutputSchema):
                    self.logger.critical(
                        f"Received unexpected LLM output: {tool_picker_result =}"
                    )
                    tool_picker_result = OutputSchema(status="failed")

        self.logger.debug(f"{tool_picker_result =}")

        return {"tool_action": tool_picker_result}

    async def _tool_caller_node(self, state: CodeAIState) -> dict:
        plan = state.plan
        current_step = plan.steps[plan.current_step]
        result: dict[str, Any] = {}

        # call tool if it is in the current state
        if current_step.tool_required:
            # TODO: retry X times if fails before marking as failed
            tool_call = state.tool_call
            assert tool_call

            tool_responses = state.tool_responses
            tool_result: CallToolResult = await self.mcp_client.call_tool_mcp(
                name=tool_call.tool, arguments=tool_call.args
            )
            tool_responses.append(tool_result)

            if tool_result.is_error:
                current_step.status = "failed"
            else:
                current_step.status = "done"

            # TODO: populate artefacts
            result["tool_responses"] = tool_responses

        plan.current_step += 1

        result["plan"] = plan
        return result

    async def _evaluator_node(self, state: CodeAIState) -> dict:
        plan = state.plan
        result = {}

        # if all steps completed
        if plan.current_step > len(plan.steps):
            plan.status = "completed"
            result["plan"] = plan

        # if any steps failed?

        return result

    async def _step_router_node(self, state: CodeAIState) -> str:
        return state.plan.status

    def _give_answer_to_user_node(self, state: CodeAIState) -> dict:
        """Return the message to the user"""
        self.logger.debug(f"{state =}")

        answer = state.message_for_user
        self.logger.info(f"Returning final answer to user: {answer}")

        return {"message_for_user": answer}

    async def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(CodeAIState)
        self.workflow.add_node("init_graph_state", self._init_graph_state_node)
        # self.workflow.add_node("summarise_history", self._summarise_history_node)
        self.workflow.add_node("goal_setter", self._goal_setter_node)
        self.workflow.add_node("planner", self._planner_node)
        self.workflow.add_node("tool_picker", self._tool_picker_node)
        self.workflow.add_node("tool_caller", self._tool_caller_node)
        self.workflow.add_node("evaluator", self._evaluator_node)
        self.workflow.add_node("step_router", self._step_router_node)
        self.workflow.add_node("give_answer_to_user", self._give_answer_to_user_node)

        self.workflow.add_edge(START, "init_graph_state")
        self.workflow.add_edge("init_graph_state", "goal_setter")
        self.workflow.add_edge("goal_setter", "planner")
        self.workflow.add_edge("planner", "tool_picker")
        self.workflow.add_edge("tool_picker", "tool_caller")

        self.workflow.add_conditional_edges(
            "evaluator",
            self._step_router_node,
            {
                "not_started": "planner",
                "in_progress": "tool_picker",
                "failed": "planner",
                "aborted": "give_answer_to_user",
                "completed": "give_answer_to_user",
            },
        )
        self.workflow.add_edge("give_answer_to_user", END)

        self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        if not os.environ.get("RUNNING_IN_DOCKER", 0):
            try:
                self.graph.get_graph().draw_mermaid_png(
                    output_file_path="code-ai-lang-graph.png"
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
                # all nodes must return dicts
                if message := state.get("message_for_user", None):
                    self.logger.info(f"User message: {message}")
                    yield message
                else:
                    self.logger.debug(f"Working in node: {node}")

    async def graph_stream(self, query: str, thread_id: str = "default_threaD"):
        """Run the graph but return the stream"""
        config = {"configurable": {"thread_id": thread_id}}

        res = await self.graph.astream({"query": query}, config=config)
        return res
