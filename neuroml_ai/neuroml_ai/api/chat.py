#!/usr/bin/env python3
"""
Chat end points

File: neuroml_ai/api/chat.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastapi import APIRouter, Request

chat_router = APIRouter()


@chat_router.post("/query")
async def query(request: Request, query: str):
    assistant = request.app.state.assistant
    return {"result": await assistant.run_graph_invoke(query)}
