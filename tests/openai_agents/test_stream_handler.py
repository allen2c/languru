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

    messages_history: typing.List[agents.items.TResponseInputItem] = []
    previous_response_id: typing.Optional[str] = None

    for user_input in [
        "Hello world",
        "What I just said?",
        "What is the time in Tokyo?",
    ]:
        runner = agents.Runner.run_streamed(
            agent,
            input=[MessageBuilder.easy_input_message(content=user_input, role="user")],
            previous_response_id=previous_response_id,
        )
        handler = OpenAIAgentsStreamHandler(runner)

        await handler.run_until_done()

        previous_response_id = runner.last_response_id

        messages_history.extend(runner.to_input_list())
