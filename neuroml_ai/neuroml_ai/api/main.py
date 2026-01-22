#!/usr/bin/env python3
"""
Main API script

File: neuroml_ai/api/main.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastmcp import Client
from neuroml_ai.api.chat import chat_router
from neuroml_ai.api.health import health_router
from gen_rag.rag import RAG


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.is_ready = False

    client_url = "http://127.0.0.1:8542/mcp"
    mcp_client = Client(client_url)

    chat_model = os.environ.get("NML_AI_CHAT_MODEL", None)
    embedding_model = os.environ.get("NML_AI_EMBEDDING_MODEL", None)

    # check that client is up
    async with mcp_client:
        await mcp_client.ping()
        tools = await mcp_client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

    # TODO: update to use main app chain, which will include the RAG
    rag = RAG(
        chat_model=chat_model, embedding_model=embedding_model
    )
    await rag.setup()

    app.state.rag = rag
    app.state.mcp = Client
    app.state.is_ready = True

    yield

    app.state.is_ready = False


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)
app.include_router(health_router)
