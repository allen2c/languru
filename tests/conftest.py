import typing

import openai
import pytest


@pytest.fixture(scope="module")
def deps_logging():
    import logging

    import logging_bullet_train

    logging_bullet_train.set_logger(logging.getLogger("languru"))
    return None


@pytest.fixture(scope="module")
def deps_logfire():
    import logfire

    import languru

    logfire.configure(
        service_name=languru.__name__ + "-tests",
        service_version=languru.__version__,
    )
    return None


@pytest.fixture(scope="module")
def instrument_openai_agents(deps_logfire: typing.Literal[None]):
    import logfire

    logfire.instrument_openai_agents()
    return None


@pytest.fixture(scope="module")
def openai_client(deps_logfire: typing.Literal[None]):
    import logfire

    _client = openai.OpenAI()
    logfire.instrument_openai(_client)
    _client.models.list()
    return _client


@pytest.fixture(scope="module")
def openai_async_client(deps_logfire: typing.Literal[None]):
    import logfire

    _client = openai.AsyncOpenAI()
    logfire.instrument_openai(_client)
    return _client
