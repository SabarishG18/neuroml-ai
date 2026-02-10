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
    """Return the input string in a sentence (testing tool only).

    This is a dummy tool used only for unit testing and debugging.
    It simply returns the input string in a formatted sentence.

    Do NOT use in production workflows - this tool provides no real functionality.

    Args:
        astring: Any string to be echoed back

    Returns:
        The input string formatted as "I got {astring}"

    Example:
        result = await dummy_code_tool("hello")
        # Returns: "I got hello"
    """
    return f"I got {astring}"


async def list_files_tool(
    path: str,
    max_depth: Optional[int] = None,
    pattern: Optional[str] = None,
    include_files: bool = True,
    include_directories: bool = True,
    recursive: bool = False,
    max_results: int = 100,
) -> Dict[str, Any]:
    """List files and directories with filtering and metadata.

    Use this tool to explore the file system structure, find specific files,
    or understand the organization of a project before making changes.

    Common use cases:
    - Find all Python files in a project: path=".", pattern="*.py", recursive=True
    - Explore directory structure before creating new files
    - Check if specific files exist
    - Understand project layout

    Do NOT use for:
    - Reading file contents (use file reading tools instead)
    - Modifying files (this tool is read-only)
    - Accessing system directories outside current workspace

    Args:
        path: Directory path to list. Must be relative to current working directory.
              Cannot contain ".." for security. Use "." for current directory.
        max_depth: Maximum directory depth to traverse. None for unlimited.
                  Only used when recursive=True.
        pattern: Glob pattern to filter results (e.g., "*.py", "*.xml", "test_*").
                 Uses Python's Path.glob matching rules.
        include_files: Whether to include files in results.
        include_directories: Whether to include directories in results.
        recursive: If True, traverse subdirectories recursively.
        max_results: Maximum number of entries to return. Prevents overwhelming
                    results from large directories.

    Returns:
        Dictionary containing:
        - files: List of file/directory entries with metadata:
            - path: Full path to the file/directory
            - type: "file", "directory", or "link"
            - modified time: Unix timestamp of last modification
            - size: Size in bytes (0 for directories)
        - error: Error message if operation failed, empty string otherwise
        - truncated: "True" if results were limited by max_results, "False" otherwise

    Examples:
        # List all Python files recursively
        result = await list_files_tool(path=".", pattern="*.py", recursive=True)

        # Explore NeuroML project structure
        result = await list_files_tool(path=".", include_directories=True, max_depth=2)

        # Find XML files in specific directory
        result = await list_files_tool(path="./models/", pattern="*.xml")

        # Quick overview of current directory
        result = await list_files_tool(path=".", max_results=20)

    Security notes:
        - Cannot access paths containing ".."
        - Limited to current working directory and subdirectories
        - Results are truncated to prevent memory issues
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

        if len(paths) > max_results:
            truncated = "True"

        for f in paths[:max_results]:
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
    """Execute Python code in a sandboxed environment.

        This is your primary tool for running Python code, testing scripts,
        and performing computational tasks. Use it to validate code before
        including it in final outputs or to generate NeuroML models programmatically.

        Common use cases:
        - Testing code snippets before finalizing them
        - Generating NeuroML models programmatically
        - Running calculations or data processing
        - Validating Python syntax
        - Importing and testing external libraries (numpy, neuroml, etc.)

        Do NOT use for:
        - System administration tasks (use command execution tools)
        - File system operations (use file management tools)
        - Network operations (no internet access)
        - Interactive input (code must be self-contained)

        Args:
            code: Complete Python code to execute. Must be valid Python syntax.
                  Can include imports, function definitions, and execution statements.
                  Cannot read user input or require interactive sessions.

        Returns:
            Dictionary with execution results:
            - stdout: Standard output from the code execution
            - stderr: Error messages and warnings
            - returncode: 0 for success, non-zero indicates errors
            - data: Additional execution metadata (timing, resource usage, etc.)

        Examples:
            # Test a simple calculation
            result = await run_python_code_tool("print(2 + 2)")
            # Expected: stdout contains "4", returncode 0

            # Generate a simple NeuroML model
            code = '''
    import neuroml
    from neuroml.utils import component_factory

    doc = component_factory(neuroml.NeuroMLDocument, id="test")
    print(f"Created document: {doc.id}")
            '''
            result = await run_python_code_tool(code)

            # Check if a library is available
            result = await run_python_code_tool(
            "import numpy; print('numpy version:', numpy.__version__)"
        )

            # Test code with error handling
            code = '''
    try:
        import nonexistent_lib
        print("Import successful")
    except ImportError as e:
        print(f"Import failed: {e}")
            '''
            result = await run_python_code_tool(code)

        Execution environment:
        - Sandboxed: Limited access to system resources
        - No internet access
        - Memory and time limits enforced
        - Common scientific libraries available (numpy, matplotlib, neuroml, etc.)
        - Working directory is preserved between calls in the same session

        Error handling:
        - Syntax errors will appear in stderr
        - Runtime errors will appear in stderr with tracebacks
        - Check returncode for success/failure status
    """
    request = RunPythonCode(code=code)
    async with sbox(".") as f:
        result = await f.run(request)
    return asdict(result)
