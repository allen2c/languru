import typing

import agents
import openai
import pytest

from languru.examples.tools import GetTimeNow
from languru.openai_chatcmpl.stream_handler import OpenAIChatCompletionStreamHandler
from languru.openai_shared.tools import base_model_to_function_tool

MODEL = "gpt-4.1-nano"


@pytest.mark.asyncio
async def test_openai_chatcmpl_stream_handler_simple(
    openai_async_client: openai.AsyncOpenAI,
):

    tools: typing.List[agents.tool.FunctionTool] = [
        base_model_to_function_tool(
            GetTimeNow.GetTimeNowArgs,
            name=GetTimeNow.name,
            description=GetTimeNow.description,
            on_invoke_tool=GetTimeNow.get_time_now,
        )
    ]  # type: ignore

    messages_history = []

    for user_input in [
        "Hello world",
        "What I just said in previous chat?",
        "What is the time in Tokyo?",
        "What is my first chat said?",
    ]:
        messages_history.append({"role": "user", "content": user_input})

        stream_handler = OpenAIChatCompletionStreamHandler(
            openai_async_client,
            messages=messages_history,
            model=MODEL,
            tools=tools,
        )

        await stream_handler.run_until_done()

        messages_history = stream_handler.get_messages_history()

        stream_handler.display_messages_history()
