#!/usr/bin/env python3
"""
Test code related tools

File: mcp_pkg/tests/test_code_tools.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport
from neuroml_mcp.server.main import create_server


@pytest.fixture()
async def mcp_client():
    mcp = await create_server()
    async with Client(transport=mcp) as mcp_client:
        yield mcp_client


@pytest.mark.asyncio
async def test_dummy_code_tool(mcp_client: Client[FastMCPTransport]):
    ret = await mcp_client.call_tool_mcp(
        "dummy_code_tool", arguments={"astring": "Hello world"}
    )
    assert ret.structuredContent["result"] == "I got Hello world"


@pytest.mark.asyncio
async def test_run_hh_simulation_tool(mcp_client):
    ret = await mcp_client.call_tool_mcp(
        "run_hh_simulation_tool",
        arguments={"current_injection": 0.1, "duration": 300.0}
    )
    import json
    result = json.loads(ret.structuredContent["stdout"])
    assert result["num_action_potentials"] > 0
    assert result["firing_rate_hz"] > 0