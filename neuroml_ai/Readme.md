---
title: NeuroML-AI
sdk: docker
app_port: 7860
---

# NeuroML-AI

AI assistant for helping with NeuroML queries and model generation.

[![GitHub CI](https://github.com/NeuroML/neuroml-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/NeuroML/neuroml-ai/actions/workflows/ci.yml)
[![GitHub](https://img.shields.io/github/license/NeuroML/neuroml-ai)](https://github.com/NeuroML/neuroml-ai/blob/master/LICENSE)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/NeuroML/neuroml-ai)](https://github.com/NeuroML/neuroml-ai/pulls)
[![GitHub issues](https://img.shields.io/github/issues/NeuroML/neuroml-ai)](https://github.com/NeuroML/neuroml-ai/issues)
[![GitHub Org's stars](https://img.shields.io/github/stars/NeuroML?style=social)](https://github.com/NeuroML)
[![Twitter Follow](https://img.shields.io/twitter/follow/NeuroML?style=social)](https://twitter.com/NeuroML)
[![Gitter](https://badges.gitter.im/NeuroML/community.svg)](https://gitter.im/NeuroML/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


![Langchain schematic](neuroml_ai/nml-ai-assistant-lang-graph.png "Langchain schematic")

Please note that this project is under active development and does not currently provide a stable release/API/ABI.

## Usage

Install the package and dependencies using `pip` or `uv pip` from the GitHub repository:

```
# in the `utils_pkg` folder:
pip install .

# in the `neuroml_ai` folder:
pip install .
```

Start the API server:

```
fastapi dev neuroml_ai/api/main.py --port 8005
```

The following environment variables need to be set:

- `NML_AI_CHAT_MODEL`: the name of the chat model to use. See below.
- `NML_AI_VS_CONFIG`: the path to the configuration file for the vector stores.
