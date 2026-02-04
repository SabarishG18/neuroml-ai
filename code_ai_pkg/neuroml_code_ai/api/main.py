#!/usr/bin/env python3
"""
Main API script

File: code_ai_pkg/neuroml_code_ai/api/main.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from neuroml_code_ai.api.chat import chat_router
from neuroml_code_ai.api.health import health_router
from neuroml_code_ai.code_ai import CodeAI
from fastmcp import Client


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.is_ready = False

    client_url = "http://127.0.0.1:8542/mcp"
    mcp_client = Client(client_url)

    code_model = os.environ.get("NML_AI_CODE_MODEL", "ollama:qwen3:1.7b")

    # check that client is up
    async with mcp_client:
        await mcp_client.ping()
        tools = await mcp_client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

    code_ai = CodeAI(code_model=code_model, memory=True, mcp_client=mcp_client)
    await code_ai.setup()

    app.state.code_ai = code_ai
    app.state.is_ready = True

    yield

    app.state.is_ready = False


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)
app.include_router(health_router)
