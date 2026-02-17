#!/usr/bin/env python3
"""
Chat end points

File: rag_pkg/gen_rag/api/chat.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import traceback

from fastapi import APIRouter, HTTPException, Request

chat_router = APIRouter()

import logging

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@chat_router.post("/query")
async def query(request: Request, query: str):
    rag = request.app.state.rag
    try:
        result = await rag.run_graph_invoke(query)
    except Exception as e:
        detail = f"{e}\n{traceback.format_exc()}"
        result = HTTPException(status_code=500, detail=detail)

        logger.error(detail)

    return {"result": result}
