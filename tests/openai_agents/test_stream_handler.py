import typing

import agents
import openai
import pytest
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from languru.examples.tools import GetTimeNow
from languru.openai_agents.messages import MessageBuilder
from languru.openai_agents.stream_handler import OpenAIAgentsStreamHandler
from languru.openai_agents.tools import base_model_to_function_tool


@pytest.mark.asyncio
async def test_openai_agents_stream_handler_simple(
    openai_async_client: openai.AsyncOpenAI,
    instrument_openai_agents: typing.Literal[None],
):
    assert instrument_openai_agents is None

    tools: typing.List[agents.tool.Tool] = [
        base_model_to_function_tool(
            GetTimeNow.GetTimeNowArgs,
            name=GetTimeNow.name,
            description=GetTimeNow.description,
            on_invoke_tool=GetTimeNow.get_time_now,
        )
    ]  # type: ignore
    agent = agents.Agent(
        name="Agent Manager",
        instructions=prompt_with_handoff_instructions("You are a concise assistant"),
        model=agents.OpenAIResponsesModel("gpt-4.1-nano", openai_async_client),
        tools=tools,
    )

    messages_history: typing.List[agents.items.TResponseInputItem] = []
    previous_response_id: typing.Optional[str] = None

    for user_input in [
        "Hello world",
        "What I just said in previous chat?",
        "What is the time in Tokyo?",
    ]:
        runner = agents.Runner.run_streamed(
            agent,
            input=[MessageBuilder.easy_input_message(content=user_input, role="user")],
            previous_response_id=previous_response_id,
        )
        handler = OpenAIAgentsStreamHandler(runner, messages_history)

        await handler.run_until_done()

        assert handler.last_response_id is not None
        previous_response_id = handler.last_response_id

    assert any(
        m.get("name") == "GetTimeNow" and m.get("type") == "function_call"
        for m in messages_history
    ), "The function call should be present in the messages history"
    assert any(
        m.get("type") == "function_call_output" for m in messages_history
    ), "The function call output should be present in the messages history"
