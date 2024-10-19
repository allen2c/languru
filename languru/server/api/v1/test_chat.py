import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def test_client():
    import languru.server.app

    with TestClient(languru.server.app.app) as client:
        yield client


@pytest.fixture
def mocked_openai_chat_completion_create():
    from openai.resources.chat.completions import Completions as OpenaiCompletions

    from languru.examples.return_values._openai import return_chat_completion

    with patch.object(
        OpenaiCompletions,
        "create",
        MagicMock(return_value=return_chat_completion),
    ):
        yield


@pytest.fixture
def mocked_openai_chat_completion_create_stream():
    from openai.resources.chat.completions import Completions as OpenaiCompletions

    from languru.examples.return_values._openai import return_chat_completion_chunks

    with patch.object(
        OpenaiCompletions,
        "create",
        MagicMock(return_value=return_chat_completion_chunks),
    ):
        yield


def test_app_chat(test_client, mocked_openai_chat_completion_create):
    chat_call = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Respond simply and concisely."},
            {"role": "user", "content": "Hello!"},
        ],
    }
    response = test_client.post("/v1/chat/completions", json=chat_call)
    print(response.text)
    assert response.status_code == 200


def test_app_chat_stream(test_client, mocked_openai_chat_completion_create_stream):
    chat_call = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "stream": True,
    }
    with test_client.stream(
        "POST", url="/v1/chat/completions", json=chat_call
    ) as response:
        answer = ""
        for line in response.iter_lines():
            line = line.replace("data:", "", 1).strip()
            if line and line != "[DONE]":
                chat_chunk = json.loads(line)
                if chat_chunk["choices"][0]["delta"]["content"]:
                    answer += chat_chunk["choices"][0]["delta"]["content"]
        assert answer
