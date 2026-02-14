#!/usr/bin/env python3
"""
Chat end points

File: rag_pkg/gen_rag/api/chat.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastapi import APIRouter, HTTPException, Request

chat_router = APIRouter()


@chat_router.post("/query")
async def query(request: Request, query: str):
    rag = request.app.state.rag
    try:
        result = await rag.run_graph_invoke(query)
    except Exception as e:
        print(e)
        result = HTTPException(status_code=500, detail=str(e))

    return {"result": result}
