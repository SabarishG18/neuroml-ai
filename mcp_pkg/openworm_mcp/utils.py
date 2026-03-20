#!/usr/bin/env python3
"""
MCP utils for OpenWorm MCP server

File: openworm_mcp/utils.py
"""

import inspect


def register_tools(mcp, modules: list):
    """Register tools from a given module

    :param modules: list of modules with tool function definitions

    """
    for module in modules:
        for fname, fn in inspect.getmembers(module, inspect.isfunction):
            if fname.endswith("_tool"):
                mcp.tool(fn)
