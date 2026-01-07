#!/usr/bin/env python3
"""
Enter one line description here.

File:

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(
    wait=wait_random_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.HTTPStatusError, httpx.ReadError, httpx.ReadTimeout)
    ),
    reraise=True,
)
async def check_api_is_ready():
    """Exponentially drop off checking that API is ready"""
    async with httpx.AsyncClient() as client:
        response = await client.get("http://127.0.0.1:8005/health/ready")
        response.raise_for_status()

        return response.json()
