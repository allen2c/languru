import typing

import agents
import openai
import pytest

from languru.examples.tools import GetTimeNow
from languru.openai_agents.messages import MessageBuilder
from languru.openai_responses.stream_handler import OpenAIResponseStreamHandler
from languru.openai_shared.tools import base_model_to_function_tool

MODEL = "gpt-4.1-nano"


@pytest.mark.asyncio
async def test_openai_responses_stream_handler_simple(
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

    rsp_id: str | None = None
    final_response = None

    for user_input in [
        "Hello world",
        "What I just said in previous chat?",
        "What is the time in Tokyo?",
        "What is my first chat said?",
    ]:
        stream_handler = OpenAIResponseStreamHandler(
            openai_async_client,
            input=[MessageBuilder.easy_input_message(content=user_input, role="user")],
            model=MODEL,
            previous_response_id=rsp_id,
            tools=tools,
        )

        await stream_handler.run_until_done()

        rsp_id = stream_handler.retrieve_final_response().id

        final_response = stream_handler.retrieve_final_response()

    assert final_response is not None
    assert len(final_response.output) > 0
    assert final_response.output[0].type == "message"
    assert len(final_response.output[0].content) > 0
    assert final_response.output[0].content[0].type == "output_text"
    assert "hello world" in final_response.output[0].content[0].text.lower()
