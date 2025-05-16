import typing

import agents
import openai
import pytest
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from languru.examples.tools import GetTimeNow
from languru.openai_agents.stream_handler import OpenAIAgentsStreamHandler
from languru.openai_agents.tools import base_model_to_function_tool


@pytest.mark.asyncio
async def test_openai_agents_stream_handler_simple(
    openai_async_client: openai.AsyncOpenAI,
):
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
        model=agents.OpenAIChatCompletionsModel("gpt-4.1-nano", openai_async_client),
        tools=tools,
    )

    input_messages: typing.List[agents.items.TResponseInputItem] = []

    for user_input in [
        "Hello world",
        "What I just said?",
        "What is the time in Tokyo?",
    ]:
        input_messages.append(
            {
                "role": "user",
                "content": [{"text": user_input, "type": "input_text"}],
            }
        )

    handler = OpenAIAgentsStreamHandler(
        agents.Runner.run_streamed(agent, input_messages)
    )

    async for event in handler.stream_events():
        print()
        print()
        print()
        print(event)
        print()
        print()
        print()
