# Welcome to Languru

The general-purpose LLM app stacks deploy AI services quickly and (stupidly) simply.

```txt
 _
| |    __ _ _ __   __ _ _   _ _ __ _   _
| |   / _` | '_ \ / _` | | | | '__| | | |
| |__| (_| | | | | (_| | |_| | |  | |_| |
|_____\__,_|_| |_|\__, |\__,_|_|   \__,_|
                  |___/
```

[![image](https://img.shields.io/pypi/v/languru.svg)](https://pypi.python.org/pypi/languru)
[![image](https://img.shields.io/pypi/l/languru.svg)](https://pypi.python.org/pypi/languru)
[![image](https://img.shields.io/pypi/pyversions/languru.svg)](https://pypi.python.org/pypi/languru)
[![PytestCI](https://github.com/dockhardman/languru/actions/workflows/python-pytest.yml/badge.svg)](https://github.com/dockhardman/languru/actions/workflows/python-pytest.yml)
[![codecov](https://codecov.io/gh/dockhardman/languru/graph/badge.svg?token=OFX6C8Z31C)](https://codecov.io/gh/dockhardman/languru)

Documentation: [Github Pages](https://dockhardman.github.io/languru/)

## Install `Languru`

```shell
pip install languru

# Install For LLM deployment.
pip install languru[all]

# Install development dependencies.
poetry install -E <extras> --with dev

# Or just install all dependencies.
poetry install -E all --with dev --with docs
```

## OpenAI Clients

Supported OpenAI clients:

- `openai.OpenAI`
- `openai.AzureOpenAI`
- `languru.openai_plugins.clients.anthropic.AnthropicOpenAI`
- `languru.openai_plugins.clients.google.GoogleOpenAI`
- `languru.openai_plugins.clients.groq.GroqOpenAI`
- `languru.openai_plugins.clients.pplx.PerplexityOpenAI`
- `languru.openai_plugins.clients.voyage.VoyageOpenAI`

## OpenAI Server

```shell
languru server run  # Remember set all needed `api-key` for OpenAI clients.
```

Query LLM service, which is fully compatible with OpenAI APIs.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8682/v1")
res = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
for choice in res.choices:
    print(f"{choice.message.role}: {choice.message.content}")
# assistant: Hello! How can I assist you today?
```

Chat streaming:

```python
client = OpenAI(base_url="http://localhost:8682/v1")
res = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    stream=True,
)
for chunk in res:
    for choice in chunk.choices:
        if choice.delta.content:
            print(choice.delta.content, end="", flush=True)
            # Hello! How can I assist you today?
```

OpenAI plugins clients:

```python
client = OpenAI(base_url="http://localhost:8682/v1")
res = client.chat.completions.create(
    model="google/gemini-1.5-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    stream=True,
)
for choice in res.choices:
    print(f"{choice.message.role}: {choice.message.content}")
```

## Concepts

- [Openai Clients](concepts/openai_clients/index.md)
- [Data Model](concepts/data_model/index.md)
- [Prompts](concepts/prompts/prompt_template.md)
- [OpenAI Server](concepts/openai_server.md)
- [Docker](concepts/docker.md)
