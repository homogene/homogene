# Homogene

[![CI](https://github.com/homogene/homogene/actions/workflows/ci.yml/badge.svg)](https://github.com/homogene/homogene/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/homogene/homogene/graph/badge.svg)](https://codecov.io/gh/homogene/homogene)
[![PyPI](https://img.shields.io/pypi/v/homogene)](https://pypi.org/project/homogene/)
[![Python](https://img.shields.io/pypi/pyversions/homogene)](https://pypi.org/project/homogene/)
[![License](https://img.shields.io/github/license/homogene/homogene)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Homogene is a Python library that helps data scientists apply LLM instructions to every row of a `pd.DataFrame` for data processing.

Running an LLM over a `pd.DataFrame` column is easy for 10 rows. At 10,000 rows, you're managing threads, catching per-row errors, and checkpointing progress. Homogene handles all of that so you focus on the instruction, not the infrastructure.

## Processors

Homogene currently ships with one processor: `Generator`, which applies a plain-text instruction to every row of a `pd.DataFrame` independently and in parallel. More processors are coming soon.

Homogene uses [LiteLLM](https://github.com/BerriAI/litellm) to route model calls. Please refer to their documentation for the full list of supported models.

## Install

```bash
pip install homogene
```

## Quick start

```python
import os
import pandas as pd
from homogene import Generator

os.environ["OPENAI_API_KEY"] = "your-api-key"

df = pd.read_csv("tickets.csv")

generator = Generator(
    df=df,
    columns=["subject", "body"],
    instruction="Identify the main topic of this support ticket in three words or less.",
    model="gpt-5.5",
)

df["topic"] = generator.run()
```

For a full walkthrough see the `Generator` [example notebook](examples/generator_example.ipynb), covering:

- Model configuration
- Freeform generation
- Structured output
- Prompt inspection
- Subset runs
- Checkpointing
- Error handling
