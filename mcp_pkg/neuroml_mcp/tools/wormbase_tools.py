#!/usr/bin/env python3
"""
Tools for querying the WormBase REST API for C. elegans biology data.

Note that docstrings here should be written for the LLM to read.

File: neuroml_mcp/tools/wormbase_tools.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
from typing import Any, Dict, Optional

import requests

WORMBASE_REST_BASE = "http://rest.wormbase.org/rest/field"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


def _get(url: str) -> Dict[str, Any]:
    """Internal helper to make a GET request and return parsed JSON or error dict."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e}", "url": url}
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to WormBase API. Check network connection.", "url": url}
    except requests.exceptions.Timeout:
        return {"error": "WormBase API request timed out.", "url": url}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "url": url}


async def query_wormbase_gene_tool(
    gene_id: str,
    field: str = "overview",
) -> Dict[str, Any]:
    """Query the WormBase REST API for information about a C. elegans gene.

    Use this tool to retrieve biological data about a specific C. elegans gene
    from the WormBase database. WormBase is the primary curated resource for
    C. elegans genetics, genomics, and biology.

    Gene IDs use the format WBGene00000000 (e.g. WBGene00000179 for unc-10).
    Common gene names (e.g. eat-4, unc-17, glr-1) can also be used as gene_id
    but WBGene IDs are more reliable.

    Inputs:

    - gene_id (str): WormBase gene identifier (e.g. "WBGene00001250") or
      common gene name (e.g. "eat-4"). Required.
    - field (str, default "overview"): which data field to retrieve.

      Available fields:
      - "overview": general gene info, name, sequence, description
      - "phenotype": phenotypes associated with loss-of-function mutants
      - "expression": tissue and cell expression data
      - "function": gene ontology (GO) terms, molecular function
      - "genetics": alleles, polymorphisms
      - "homology": orthologues and paralogues in other species
      - "interactions": genetic and physical interactions with other genes
      - "references": published papers about this gene

    Output:

    Dictionary containing the requested field data from WormBase.
    The structure varies by field but always includes either:
    - The requested biological data as nested dicts/lists
    - An "error" key with an error message if the query failed

    Examples:

    - Get overview of eat-4 gene: query_wormbase_gene_tool(gene_id="eat-4")
    - Get phenotypes for unc-17: query_wormbase_gene_tool(gene_id="unc-17", field="phenotype")
    - Get expression pattern of glr-1: query_wormbase_gene_tool(gene_id="glr-1", field="expression")
    - Get gene interactions: query_wormbase_gene_tool(gene_id="WBGene00001250", field="interactions")
    - Get published papers: query_wormbase_gene_tool(gene_id="eat-4", field="references")
    """
    url = f"{WORMBASE_REST_BASE}/gene/{gene_id}/{field}"
    result = _get(url)
    return result


async def query_wormbase_neuron_tool(
    neuron_name: str,
    field: str = "overview",
) -> Dict[str, Any]:
    """Query the WormBase REST API for information about a specific C. elegans neuron.

    Use this tool to retrieve anatomical and functional data about a specific
    neuron or neuron class in C. elegans from WormBase. C. elegans has exactly
    302 neurons with well-characterised identities.

    Neuron names follow standard C. elegans nomenclature, e.g.:
    - AWC, AWA, AWB (chemosensory neurons)
    - AIY, AIZ, AIA (interneurons)
    - DB1-DB7, VB1-VB11 (motor neurons)
    - AVAL, AVAR, AVBL, AVBR (command interneurons)

    Inputs:

    - neuron_name (str): standard C. elegans neuron or anatomy term name.
      Examples: "ADAL", "AWC", "AIY", "DB1", "RIA". Required.
    - field (str, default "overview"): which data field to retrieve.

      Available fields:
      - "overview": general description, lineage, position
      - "anatomy_function": known functions of this neuron
      - "expressed_in": genes expressed in this neuron
      - "innervates": synaptic partners and connectivity
      - "references": published papers about this neuron

    Output:

    Dictionary containing the requested field data from WormBase.
    Always includes either the requested data or an "error" key.

    Examples:

    - Get overview of AWC neuron: query_wormbase_neuron_tool(neuron_name="AWCON")
    - Get function of AIY: query_wormbase_neuron_tool(neuron_name="AIYL", field="anatomy_function")
    - Get genes expressed in RIA: query_wormbase_neuron_tool(neuron_name="RIAL", field="expressed_in")
    """
    url = f"{WORMBASE_REST_BASE}/anatomy_term/{neuron_name}/{field}"
    result = _get(url)
    return result


async def query_wormbase_phenotype_tool(
    phenotype_id: str,
) -> Dict[str, Any]:
    """Query the WormBase REST API for information about a C. elegans phenotype.

    Use this tool to look up a specific phenotype in WormBase and retrieve
    which genes are associated with it, along with descriptions.

    Phenotype IDs use the format WBPhenotype:0000000.
    Common phenotype terms include things like:
    - "uncoordinated" (Unc phenotype)
    - "paralysed"
    - "slow growth"
    - "embryonic lethal"

    Inputs:

    - phenotype_id (str): WormBase phenotype identifier in format
      "WBPhenotype:0000000" or a descriptive term like "uncoordinated". Required.

    Output:

    Dictionary with phenotype overview data including:
    - Associated genes
    - Phenotype description
    - Related phenotypes
    Or an "error" key if the query failed.

    Examples:

    - Look up uncoordinated phenotype: query_wormbase_phenotype_tool(phenotype_id="WBPhenotype:0000643")
    - Look up paralysed phenotype: query_wormbase_phenotype_tool(phenotype_id="WBPhenotype:0000475")
    """
    url = f"{WORMBASE_REST_BASE}/phenotype/{phenotype_id}/overview"
    result = _get(url)
    return result


async def search_wormbase_tool(
    query: str,
    entity_type: str = "gene",
    limit: int = 5,
) -> Dict[str, Any]:
    """Search WormBase for C. elegans genes, neurons, or other biological entities by name or keyword.

    Use this tool when you have a gene name, partial name, or keyword and need
    to find the corresponding WormBase identifier or confirm an entity exists.
    This is useful before calling query_wormbase_gene_tool or query_wormbase_neuron_tool
    when you are unsure of the exact identifier.

    Inputs:

    - query (str): search term, e.g. a gene name, keyword, or partial name. Required.
      Examples: "eat-4", "glutamate", "acetylcholine", "chemosensory"
    - entity_type (str, default "gene"): type of entity to search for.
      Options: "gene", "anatomy", "phenotype", "protein", "variation"
    - limit (int, default 5): maximum number of results to return (1-20).

    Output:

    Dictionary with keys:
    - hits (list): list of matching entities, each with name and WormBase ID
    - total (int): total number of matches found
    - error (str): error message if query failed

    Examples:

    - Search for eat-4 gene: search_wormbase_tool(query="eat-4")
    - Search for glutamate genes: search_wormbase_tool(query="glutamate", entity_type="gene")
    - Search for chemosensory neurons: search_wormbase_tool(query="chemosensory", entity_type="anatomy")
    - Find uncoordinated phenotypes: search_wormbase_tool(query="uncoordinated", entity_type="phenotype")
    """
    try:
        url = f"http://rest.wormbase.org/rest/search/{entity_type}/{requests.utils.quote(query)}"
        params = {"limit": limit}
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e}", "hits": [], "total": 0}
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to WormBase API.", "hits": [], "total": 0}
    except requests.exceptions.Timeout:
        return {"error": "WormBase API request timed out.", "hits": [], "total": 0}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "hits": [], "total": 0}