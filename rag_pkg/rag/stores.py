#!/usr/bin/env python3
"""
Vector stores

File: gen_rag/stores.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import sys
from glob import glob
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .utils import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
    setup_embedding,
)

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class Vector_Stores(object):
    """Vector stores"""

    def __init__(
        self,
        embedding_model: str,
        logging_level: int = logging.DEBUG,
        domains: list[str] = ["nml"],
    ):
        """Init"""
        # per store
        self.default_k = 5
        self.k_max = 10
        self.k = self.default_k
        self.sim_thresh = 0.15
        self.embedding_model = embedding_model
        self.embeddings = None

        # we prefer markdown because the one page PDF that is available for the
        # documentation does not work too well with embeddings
        my_path = Path(__file__).parent
        self.data_dir = f"{my_path}/data/"
        self.stores_path = f"{self.data_dir}/vector-stores"

        self.text_vector_stores: dict[str, dict[str, Chroma]] = {}
        self.image_vector_stores: dict[str, Any] = {}

        self.logger = logging.getLogger("NeuroML-AI")
        self.logger.setLevel(logging_level)
        self.logger.propagate = False

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(LoggerInfoFilter())
        stdout_handler.setFormatter(logger_formatter_info)
        self.logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging_level)
        stderr_handler.addFilter(LoggerNotInfoFilter())
        stderr_handler.setFormatter(logger_formatter_other)
        self.logger.addHandler(stderr_handler)

    def setup(self):
        """Setup embeddings"""
        self.embeddings = setup_embedding(self.embedding_model, self.logger)
        # extract model name
        if self.embedding_model.lower().startswith("huggingface:"):
            # strip suffix/prefix
            self.embedding_model = (
                self.embedding_model.replace("huggingface:", "")
                .replace(":cheapest", "")
                .replace(":fastest", "")
            )
            # strip collection name
            splits = self.embedding_model.split("/")
            self.embedding_model = "".join(splits[1:])
        elif self.embedding_model.lower().startswith("ollama:"):
            # strip prefix
            self.embedding_model = self.embedding_model.replace("ollama:", "")

    def inc_k(self, inc: int = 1):
        """Increase k by inc

        :param inc: int to increase k by
        :returns: True if k was increased, False otherwise

        """
        if (self.k + inc) <= self.k_max:
            self.k += inc
            self.logger.debug(f"k increased to {self.k =}")
            return True

        return False

    def reset_k(self):
        """Reset k to default value"""
        self.k = self.default_k
        self.logger.debug(f"k reset to {self.k =}")

    def load(self, domain: str):
        """Create/load the vector store"""
        assert self.embeddings

        self.logger.debug("Setting up/loading Chroma vector store")

        self.logger.debug(f"{self.stores_path =}")
        vector_stores = glob(f"{self.stores_path}/{domain}-*", recursive=False)
        self.logger.debug(f"{vector_stores =}")

        assert len(vector_stores)

        self.text_vector_stores[domain] = {}

        for store in vector_stores:
            store_path = Path(store)

            assert store_path.is_dir()

            vs_persist_dir = (
                f"{store_path.name}_{self.embedding_model.replace(':', '_')}.db"
            )
            self.logger.debug(f"{vs_persist_dir =}")

            chroma_client_settings_text = chromadb.config.Settings(
                is_persistent=True,
                persist_directory=vs_persist_dir,
                anonymized_telemetry=False,
            )
            store = Chroma(
                collection_name=store_path.name,
                embedding_function=self.embeddings,
                client_settings=chroma_client_settings_text,
            )

            self.text_vector_stores[domain][store_path.name] = store

    def retrieve(self, domain: str, query: str) -> list[tuple[Document, float]]:
        """Retrieve embeddings from documentation to answer a query

        :param query: user query
        :returns: list of tuples (document, score)

        """
        self.load(domain)

        assert len(self.text_vector_stores[domain])
        stores = self.text_vector_stores[domain]

        res = []

        for sname, store in stores.items():
            data = store.similarity_search_with_relevance_scores(
                query, k=self.k, score_threshold=self.sim_thresh
            )
            self.logger.debug(f"{data =}")
            res.extend(data)

        return res
