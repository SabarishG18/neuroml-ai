#!/usr/bin/env python3
"""
General code execution tools.

Note that docstrings here should be written for the LLM to read.

File: neuroml_mcp/tools/code_tools.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any, Dict, List
from dataclasses import asdict

from neuroml_mcp.tools.sandbox import docker, local
from neuroml_mcp.tools.sandbox.sandbox import RunCommand, RunPythonCode

# set the implementation for development
sbox = local.LocalSandbox


async def dummy_code_tool(astring: str) -> str:
    """Dummy tool that returns the string given to it. Doesn't do anything
    else. Only here for unit testing. Ignore me.

    :param astring: a string
    :type astring: str
    :returns: the given string in a sentence
    :rtype: str

    """
    return f"I got {astring}"


async def run_command_tool(command: str) -> Dict[str, Any]:
    """Runs a command in a shell.

    Input:

    - command (string):
      Example: "ls -l -A"

    Output:

    Dictionary with keys:
    - stdout (str)
    - stderr (str)
    - returncode (int)
    - data (dict): additional metadata

    Examples:
    - run_command_tool(command="ls")
    - run_command_tool(command="ls -l"])

    """
    request = RunCommand(command=command.split())
    async with sbox(".") as f:
        result = await f.run(request)
    return asdict(result)


async def run_python_code_tool(code: str) -> Dict[str, Any]:
    """Run given Python code

    Input:

    - code (strings): Python code as a string

    Output:

    Dictionary with keys:
    - stdout (str)
    - stderr (str)
    - returncode (int)
    - data (dict): additional metadata

    Example:
    - run_python_code_tool(code="import numpy")

    """
    request = RunPythonCode(code=code)
    async with sbox(".") as f:
        result = await f.run(request)
    return asdict(result)
