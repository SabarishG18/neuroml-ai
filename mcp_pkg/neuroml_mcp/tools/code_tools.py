#!/usr/bin/env python3
"""
General code execution tools.

Note that docstrings here should be written for the LLM to read.

File: neuroml_mcp/tools/code_tools.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuroml_mcp.tools.sandbox.sandbox import RunPythonCode

from .sandbox import nml_mcp_sandbox

# set the implementation for development
sbox = nml_mcp_sandbox


async def dummy_code_tool(astring: str) -> str:
    """Dummy tool that returns the string given to it. Doesn't do anything
    else. Only here for unit testing. Ignore me.

    :param astring: a string
    :type astring: str
    :returns: the given string in a sentence
    :rtype: str

    """
    return f"I got {astring}"


async def list_files_tool(
    path: str,
    max_depth: Optional[int] = None,
    pattern: Optional[str] = None,
    include_files: bool = True,
    include_directories: bool = True,
    recursive: bool = False,
    limit: int = 100,
) -> Dict[str, Any]:
    """List files and directories in a given path

    Input:

    - a path (string): path relative to current root. Must not include ".."
    - max_depth (integer): maximum number of levels
    - pattern (string): pattern to match, passed to Python's Path.glob function.
    - include_files (bool, default True): include files in list
    - include_directories (bool, default True): include directories in list
    - recursive (bool, default False): if True, traverse subdirectories
    - limit (integer, default 100): maximum number of entries returned

    Output:

    Dictionary with keys:

    - files (list of dictionaries): list of returned files with metadata:
        - path: full file path
        - type: whether it is a file or a directory
        - modified time: time file was last modified
        - size: size of file
    - error (str): error if any
    - truncated (bool): True if list was limited to `limit` entries

    Examples:

    - list all files and folders in current directory: list_files_tool(path=".")
    - list only python files in "source" directory: list_files_tool(path="./source/", pattern="*.py")

    """
    the_path = Path(path)
    if pattern is None:
        pattern = "*"

    truncated = "False"
    error = ""
    files: List[Dict[str, Any]] = []
    paths: List[Path] = []

    if ".." in path:
        return {
            "files": [],
            "truncated": "False",
            "error": "Path contains '..', exiting.",
        }

    try:
        if recursive:
            paths = list(the_path.rglob(pattern))
        else:
            paths = list(the_path.glob(pattern))

        if len(paths) > limit:
            truncated = "True"

        for f in paths[:limit]:
            ftype = "file"
            if f.is_dir():
                ftype = "directory"
            if f.is_symlink():
                ftype = "link"
            files.append(
                {
                    "path": str(f),
                    "type": ftype,
                    "modified time": f.stat().st_mtime,
                    "size": f.stat().st_size,
                }
            )
    except Exception as e:
        error = e.__str__()

    result = {"files": files, "error": error, "truncated": truncated}

    return result


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
