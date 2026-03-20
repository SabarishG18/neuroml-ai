#!/usr/bin/env python3
"""
MCP server for OpenWorm tools (HH simulation, WormBase queries)

This server is separate from the NeuroML MCP server.
NeuroML MCP handles NeuroML-specific tools (code gen, LEMS, NeuroML models).
OpenWorm MCP handles organism-specific tools (HH simulations, WormBase data).

File: openworm_mcp/server/main.py
"""

from openworm_mcp.utils import register_tools
from openworm_mcp.tools import hh_tools, wormbase_tools
from textwrap import dedent
from fastmcp import FastMCP
from fastmcp_docs import FastMCPDocs
from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse
import asyncio


async def create_server():
    """main server creator"""
    usage = dedent(
        """
        OpenWorm assistant server.

        Provides tools for:
        - Hodgkin-Huxley neuron simulations (based on openworm/hodgkin_huxley_tutorial)
        - WormBase REST API queries for C. elegans biology data
        """
    )
    mcp = FastMCP("openworm_MCP", instructions=usage, port=8543)
    register_tools(mcp, [hh_tools, wormbase_tools])

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> PlainTextResponse:
        return PlainTextResponse("OK")

    @mcp.custom_route("/list", methods=["GET"])
    async def tool_list(request: Request) -> JSONResponse:
        all_tools = await (mcp.get_tools())
        tools_description = [{str(tool.name): str(tool.description)} for name, tool in all_tools.items()]
        resp = {"registered_tools": tools_description}
        return JSONResponse(resp)

    docs = FastMCPDocs(mcp, title="OpenWorm MCP")
    await docs.setup()

    return mcp

def main():
    """main runner method"""
    mcp = asyncio.run(create_server())
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
