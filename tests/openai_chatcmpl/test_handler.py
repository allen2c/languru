import base64
import pathlib
import typing

import agents
import openai
import pytest
from openai.types.chat.chat_completion import ChatCompletion

from languru.examples.tools import GetTimeNow
from languru.openai_chatcmpl.handler import OpenAIChatCompletionHandler
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
    final_chatcmpl: ChatCompletion

    for user_input in [
        "Hello world",
        "What I just said in previous chat?",
        "What is the time in Tokyo?",
        "What is my first chat said?",
    ]:
        messages_history.append({"role": "user", "content": user_input})

        handler = OpenAIChatCompletionHandler(
            openai_async_client,
            messages=messages_history,
            model=MODEL,
            tools=tools,
        )

        await handler.run_until_done()

        final_chatcmpl = handler.retrieve_last_chatcmpl()
        assert final_chatcmpl.choices
        assert final_chatcmpl.choices[0].message.content

        messages_history = handler.get_messages_history()

        handler.display_messages_history()

    assert "final_chatcmpl" in locals()
    assert final_chatcmpl.choices[0].message.content is not None
    assert "hello world" in final_chatcmpl.choices[0].message.content.lower()


@pytest.mark.asyncio
async def test_openai_chatcmpl_handler_audio(
    openai_async_client: openai.AsyncOpenAI,
):
    data_dir = pathlib.Path("data")
    data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    user_audio_filepath = data_dir.joinpath("user_says.wav")
    bot_audio_filepath = data_dir.joinpath("bot_says.wav")

    if user_audio_filepath.is_file() is False:
        async with openai_async_client.audio.speech.with_streaming_response.create(
            input="Who are you?",
            model="gpt-4o-mini-tts",
            voice="alloy",
            response_format="wav",
            instructions="Speak in a cheerful and positive tone.",
        ) as response:
            await response.stream_to_file(user_audio_filepath)

    handler = OpenAIChatCompletionHandler(
        openai_async_client,
        model="gpt-4o-mini-audio-preview",
        modalities=["text", "audio"],  # want both text & speech out
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "system",
                "content": "You are a concise assistant, speak very very fast (speed x2)",  # noqa: E501
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(
                                user_audio_filepath.read_bytes()
                            ).decode("ascii"),
                            "format": "wav",
                        },
                    },
                ],
            },
        ],
    )

    await handler.run_until_done()

    chatcmpl = handler.retrieve_last_chatcmpl()

    assert chatcmpl.choices[0].message.audio is not None
    audio_b64 = chatcmpl.choices[0].message.audio.data
    audio_bytes = base64.b64decode(audio_b64)
    bot_audio_filepath.write_bytes(audio_bytes)
