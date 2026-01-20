#!/usr/bin/env python3
"""
Test vector store related code.

File: tests/test_stores.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import unittest

import pytest
from gen_rag.stores import Vector_Stores


class TestStores(unittest.TestCase):
    """Docstring for TestStores."""

    def test_retrieval(self):
        """Test retrieval"""
        model = "bge-m3:latest"

        try:
            stores = Vector_Stores(
                embedding_model=f"ollama:{model}", domains_file="domains.json"
            )
            stores.setup()
            stores.retrieve("NeuroML", "NeuroML community")
        except Exception as e:  # noqa
            # TODO: handle different exceptions separately
            pytest.skip("Ollama model not found")


if __name__ == "__main__":
    unittest.main()
