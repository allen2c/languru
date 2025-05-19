import copy
import logging
import typing

import agents
import httpx
import logfire
import openai
import pydantic
from agents.run_context import RunContextWrapper, TContext
from agents.usage import Usage
from openai._streaming import AsyncStream
from openai._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from openai.types.chat import (
    chat_completion,
    chat_completion_chunk,
    completion_create_params,
)
from openai.types.chat.chat_completion_audio import ChatCompletionAudio
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_prediction_content_param import (
    ChatCompletionPredictionContentParam,
)
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.shared.chat_model import ChatModel
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.metadata import Metadata
from str_or_none import str_or_none

from languru.openai_shared.messages import sanitize_chatcmpl_messages_input
from languru.openai_shared.tools import function_tool_to_chatcmpl_tool_param

logger = logging.getLogger(__name__)

FAKE_ID = "__fake_id__"
FAKE_MODEL = "__fake_model__"


class DeltaAudio(pydantic.BaseModel):
    id: typing.Optional[str] = None
    transcript: typing.Optional[str] = None
    data: typing.Optional[str] = None
    expires_at: typing.Optional[int] = None


class OpenAIChatCompletionStreamHandler(typing.Generic[TContext]):
    def __init__(
        self,
        openai_client: openai.AsyncOpenAI,
        *,
        messages: typing.Iterable[ChatCompletionMessageParam],
        model: typing.Union[str, ChatModel],
        stream: typing.Literal[True] = True,
        audio: typing.Optional[ChatCompletionAudioParam] | NotGiven = NOT_GIVEN,
        frequency_penalty: typing.Optional[float] | NotGiven = NOT_GIVEN,
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: (
            typing.Iterable[completion_create_params.Function] | NotGiven
        ) = NOT_GIVEN,
        logit_bias: typing.Optional[typing.Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: typing.Optional[bool] | NotGiven = NOT_GIVEN,
        max_completion_tokens: typing.Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: typing.Optional[int] | NotGiven = NOT_GIVEN,
        metadata: typing.Optional[Metadata] | NotGiven = NOT_GIVEN,
        modalities: (
            typing.Optional[typing.List[typing.Literal["text", "audio"]]] | NotGiven
        ) = NOT_GIVEN,
        n: typing.Optional[int] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        prediction: (
            typing.Optional[ChatCompletionPredictionContentParam] | NotGiven
        ) = NOT_GIVEN,
        presence_penalty: typing.Optional[float] | NotGiven = NOT_GIVEN,
        reasoning_effort: typing.Optional[ReasoningEffort] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: typing.Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: (
            typing.Optional[typing.Literal["auto", "default", "flex"]] | NotGiven
        ) = NOT_GIVEN,
        stop: (
            typing.Union[typing.Optional[str], typing.List[str], None] | NotGiven
        ) = NOT_GIVEN,
        store: typing.Optional[bool] | NotGiven = NOT_GIVEN,
        stream_options: (
            typing.Optional[ChatCompletionStreamOptionsParam] | NotGiven
        ) = NOT_GIVEN,
        temperature: typing.Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: (
            typing.Optional[ChatCompletionToolChoiceOptionParam] | NotGiven
        ) = NOT_GIVEN,
        tools: typing.List[agents.FunctionTool] | NotGiven = NOT_GIVEN,
        top_logprobs: typing.Optional[int] | NotGiven = NOT_GIVEN,
        top_p: typing.Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        web_search_options: (
            completion_create_params.WebSearchOptions | NotGiven
        ) = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        context: TContext | None = None,
        **kwargs,
    ):
        self.__openai_client = openai_client
        self.__messages = messages
        self.__model = model
        self.__stream = stream
        self.__audio = audio
        self.__frequency_penalty = frequency_penalty
        self.__function_call = function_call
        self.__functions = functions
        self.__logit_bias = logit_bias
        self.__logprobs = logprobs
        self.__max_completion_tokens = max_completion_tokens
        self.__max_tokens = max_tokens
        self.__metadata = metadata
        self.__modalities = modalities
        self.__n = n
        self.__parallel_tool_calls = parallel_tool_calls
        self.__prediction = prediction
        self.__presence_penalty = presence_penalty
        self.__reasoning_effort = reasoning_effort
        self.__response_format = response_format
        self.__seed = seed
        self.__service_tier = service_tier
        self.__stop = stop
        self.__store = store
        self.__stream_options = (
            {"include_usage": True}
            if stream_options is NOT_GIVEN or stream_options is None
            else stream_options
        )
        self.__temperature = temperature
        self.__tool_choice = tool_choice
        self.__tools = tools
        self.__top_logprobs = top_logprobs
        self.__top_p = top_p
        self.__user = user
        self.__web_search_options = web_search_options
        self.__extra_headers = extra_headers
        self.__extra_query = extra_query
        self.__extra_body = extra_body
        self.__timeout = timeout

        self.__chatcmpls: typing.List[chat_completion.ChatCompletion] = []
        self.__messages_history: typing.List[ChatCompletionMessageParam] = [
            copy.deepcopy(m) for m in messages
        ]
        self.__context = context
        self.__accumulated_usage = Usage()

    async def run_until_done(self, *, limit: int = 10) -> None:
        if limit <= 0 or limit > 25:
            raise ValueError("Limit must be between 1 and 25")

        current_limit = 0
        required_tool_call = True

        from rich import print

        print(sanitize_chatcmpl_messages_input(self.__messages_history))

        while required_tool_call and current_limit <= limit:
            stream = await self.__openai_client.chat.completions.create(
                messages=sanitize_chatcmpl_messages_input(self.__messages_history),
                model=self.__model,
                stream=True,
                audio=self.__audio,
                frequency_penalty=self.__frequency_penalty,
                function_call=self.__function_call,  # type: ignore
                functions=self.__functions,
                logit_bias=self.__logit_bias,
                logprobs=self.__logprobs,
                max_completion_tokens=self.__max_completion_tokens,
                max_tokens=self.__max_tokens,
                metadata=self.__metadata,
                modalities=self.__modalities,
                n=self.__n,
                parallel_tool_calls=self.__parallel_tool_calls,
                prediction=self.__prediction,
                presence_penalty=self.__presence_penalty,
                reasoning_effort=self.__reasoning_effort,  # type: ignore
                response_format=self.__response_format,
                seed=self.__seed,
                service_tier=self.__service_tier,  # type: ignore
                stop=self.__stop,
                store=self.__store,
                stream_options=self.__stream_options,  # type: ignore
                temperature=self.__temperature,
                tool_choice=self.__tool_choice,  # type: ignore
                tools=(
                    [
                        function_tool_to_chatcmpl_tool_param(tool)
                        for tool in self.__tools
                    ]
                    if self.__tools and self.__tools != NOT_GIVEN
                    else NOT_GIVEN
                ),
                top_logprobs=self.__top_logprobs,
                top_p=self.__top_p,
                user=self.__user,
                web_search_options=self.__web_search_options,
                extra_headers=self.__extra_headers,
                extra_query=self.__extra_query,
                extra_body=self.__extra_body,
                timeout=self.__timeout,
            )
            stream = typing.cast(
                AsyncStream[chat_completion_chunk.ChatCompletionChunk], stream
            )

            _chatcmpl = chat_completion.ChatCompletion(
                id=FAKE_ID,
                choices=[
                    chat_completion.Choice(
                        finish_reason="stop",  # placeholder
                        index=0,
                        logprobs=None,  # placeholder
                        message=ChatCompletionMessage(role="assistant"),
                    )
                ],
                created=0,
                model=FAKE_MODEL,
                object="chat.completion",
                service_tier=None,
                system_fingerprint=None,
                usage=None,
            )
            self.__chatcmpls.append(_chatcmpl)

            # Stream
            async for chunk in stream:
                self.__update_chatcmpl_from_chunk(_chatcmpl, chunk)

                await self.on_chatcmpl_chunk(chunk)

            await self.on_chatcmpl_done(_chatcmpl)

            self.__messages_history.append(
                _chatcmpl.choices[0].message.model_dump(
                    exclude_none=True,
                )  # type: ignore
            )

            self.__accumulated_usage.add(
                Usage(
                    requests=1,
                    input_tokens=(
                        _chatcmpl.usage.prompt_tokens if _chatcmpl.usage else 0
                    ),
                    output_tokens=(
                        _chatcmpl.usage.completion_tokens if _chatcmpl.usage else 0
                    ),
                    total_tokens=_chatcmpl.usage.total_tokens if _chatcmpl.usage else 0,
                )
            )

            required_tool_call = (
                _chatcmpl.choices[0].message.tool_calls is not None
                and len(_chatcmpl.choices[0].message.tool_calls) > 0
            )
            logger.debug(f"required_tool_call: {required_tool_call}")
            if required_tool_call:
                # Handle tool call
                for _tool_call in _chatcmpl.choices[0].message.tool_calls or []:
                    tool_call_output = await self.execute_chatcmpl_tool_call(_tool_call)
                    self.__messages_history.append(tool_call_output)

        return None

    def retrieve_last_chatcmpl(self) -> chat_completion.ChatCompletion:
        if not self.__chatcmpls:
            raise ValueError("No any chat completion available")
        return self.__chatcmpls[-1].model_copy(deep=True)

    def get_messages_history(self) -> typing.List[ChatCompletionMessageParam]:
        return copy.deepcopy(self.__messages_history)

    def get_usage(self) -> Usage:
        return self.__accumulated_usage

    async def execute_chatcmpl_tool_call(
        self, chatcmpl_tool_call: ChatCompletionMessageToolCall
    ) -> ChatCompletionToolMessageParam:
        with logfire.span(
            f"execute_chatcmpl_tool_call:{chatcmpl_tool_call.id}"
        ) as span:
            tool_msg = ChatCompletionToolMessageParam(
                tool_call_id=chatcmpl_tool_call.id,
                role="tool",
                content="",
            )
            tool: agents.FunctionTool | None = next(
                (
                    tool
                    for tool in (
                        self.__tools
                        if self.__tools and not isinstance(self.__tools, NotGiven)
                        else []
                    )
                    if tool.name == chatcmpl_tool_call.function.name
                ),
                None,
            )

            if tool is None:
                logger.error(f"Tool not found: {chatcmpl_tool_call.function.name}")
                span.set_attributes({"success": False, "error": "tool_not_found"})
                return (
                    tool_msg.update({"content": "Current tool call is not supported"})
                    or tool_msg
                )

            try:
                tool_output = await tool.on_invoke_tool(
                    RunContextWrapper(self.__context, self.__accumulated_usage),
                    chatcmpl_tool_call.function.arguments,
                )
                span.set_attributes({"success": True})
                return tool_msg.update({"content": tool_output}) or tool_msg

            except Exception as e:
                logger.error(f"Error executing tool: {e}")
                span.set_attributes({"success": False, "error": "tool_execution_error"})
                return (
                    tool_msg.update({"content": "Error, please try again later"})
                    or tool_msg
                )

    async def on_chatcmpl_chunk(
        self, chatcmpl_chunk: chat_completion_chunk.ChatCompletionChunk
    ) -> None:
        pass

    async def on_chatcmpl_done(self, chatcmpl: chat_completion.ChatCompletion) -> None:
        pass

    def __update_chatcmpl_from_chunk(
        self,
        chatcmpl: chat_completion.ChatCompletion,
        chatcmpl_chunk: chat_completion_chunk.ChatCompletionChunk,
    ) -> None:
        # Update the chatcmpl
        if chatcmpl.id == FAKE_ID:
            chatcmpl.id = chatcmpl_chunk.id

        if chatcmpl.created == 0:
            chatcmpl.created = chatcmpl_chunk.created

        if chatcmpl.model == FAKE_MODEL:
            chatcmpl.model = chatcmpl_chunk.model

        if chatcmpl.service_tier is None and chatcmpl_chunk.service_tier is not None:
            chatcmpl.service_tier = chatcmpl_chunk.service_tier

        if (
            chatcmpl.system_fingerprint is None
            and chatcmpl_chunk.system_fingerprint is not None
        ):
            chatcmpl.system_fingerprint = chatcmpl_chunk.system_fingerprint

        if chatcmpl.usage is None and chatcmpl_chunk.usage is not None:
            chatcmpl.usage = chatcmpl_chunk.usage

        # Update the choice
        assert len(chatcmpl.choices) > 0
        if chatcmpl_chunk.choices is None or len(chatcmpl_chunk.choices) == 0:
            return

        chatcmpl_choice = chatcmpl.choices[0]
        chunk_choice = chatcmpl_chunk.choices[0]

        if chunk_choice.finish_reason is not None:
            chatcmpl_choice.finish_reason = chunk_choice.finish_reason

        if chunk_choice.logprobs is not None:
            chatcmpl_choice.logprobs = (
                chat_completion.ChoiceLogprobs.model_validate_json(
                    chunk_choice.logprobs.model_dump_json()
                )
            )

        if chunk_choice.delta.content is not None:  # message delta
            if chatcmpl_choice.message.content is None:
                chatcmpl_choice.message.content = ""
            chatcmpl_choice.message.content += chunk_choice.delta.content

        if delta_audio_data := getattr(chunk_choice.delta, "audio", None):  # audio
            delta_audio = DeltaAudio.model_validate(delta_audio_data)
            if chatcmpl_choice.message.audio is None:
                chatcmpl_choice.message.audio = ChatCompletionAudio(
                    id=FAKE_ID, data="", expires_at=0, transcript=""
                )
            if (
                delta_audio.id is not None
                and chatcmpl_choice.message.audio.id == FAKE_ID
            ):
                chatcmpl_choice.message.audio.id = delta_audio.id
            if delta_audio.transcript is not None:
                chatcmpl_choice.message.audio.transcript += delta_audio.transcript
            if delta_audio.data is not None:
                chatcmpl_choice.message.audio.data += delta_audio.data
            if (
                delta_audio.expires_at is not None
                and chatcmpl_choice.message.audio.expires_at == 0
            ):
                chatcmpl_choice.message.audio.expires_at = delta_audio.expires_at

        if chunk_choice.delta.tool_calls is not None:
            chunk_tool_call = chunk_choice.delta.tool_calls[0]
            if chatcmpl_choice.message.tool_calls is None:
                chatcmpl_choice.message.tool_calls = []
            if (chunk_tool_call.index + 1) > len(chatcmpl_choice.message.tool_calls):
                chatcmpl_choice.message.tool_calls.append(
                    ChatCompletionMessageToolCall.model_validate(
                        {
                            "id": FAKE_ID,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )
                )

            chatcmpl_choice_message_tool_call = chatcmpl_choice.message.tool_calls[
                chunk_tool_call.index
            ]
            if (
                chunk_tool_call.id is not None
                and chatcmpl_choice_message_tool_call.id == FAKE_ID
            ):
                chatcmpl_choice_message_tool_call.id = chunk_tool_call.id
            if chunk_tool_call.function is not None:
                if chunk_tool_call.function.name is not None:
                    chatcmpl_choice_message_tool_call.function.name += (
                        chunk_tool_call.function.name
                    )
                if chunk_tool_call.function.arguments is not None:
                    chatcmpl_choice_message_tool_call.function.arguments += (
                        chunk_tool_call.function.arguments
                    )

        return None

    def display_messages_history(self) -> None:
        for message in self.__messages_history:
            _role = message["role"]
            _content = message.get("content", None)
            tool_calls = message.get("tool_calls", None)
            tool_call_id = message.get("tool_call_id", None)

            if _content is not None:
                if isinstance(_content, str):
                    if tool_call_id is not None:
                        print(f"\n{_role:10s}: [{tool_call_id[:10]}] {_content}")
                    else:
                        print(f"\n{_role:10s}: {_content}")
                else:
                    for _c in _content:
                        if _c["type"] == "text":
                            _value = _c["text"]
                        elif _c["type"] == "image_url":
                            _value = _c["image_url"]["url"]
                        elif _c["type"] == "input_audio":
                            _value = (
                                (_c["input_audio"]["data"][:100] + "...")
                                if len(_c["input_audio"]["data"]) > 100
                                else _c["input_audio"]["data"]
                            )
                        elif _c["type"] == "file":
                            _value = ""
                            _file_id = _c["file"].get("file_id", None)
                            _filename = _c["file"].get("filename", None)
                            _file_data = _c["file"].get("file_data", None)
                            if _file_id is not None:
                                _value += f"({_file_id})"
                            if _filename is not None:
                                _value += f"({_filename})"
                            if _file_data is not None:
                                _value += f"({_file_data[:100] + '...'})"
                        else:
                            _value = str(_c)
                        print(f"\n{_role:10s}: {_value}")

            if message.get("audio", None):
                _audio_id = str_or_none(message["audio"].get("id", None))  # type: ignore  # noqa: E501
                _audio_data = str_or_none(message["audio"].get("data", None))  # type: ignore  # noqa: E501
                _audio_transcript = str_or_none(
                    message["audio"].get("transcript", None)  # type: ignore
                )
                print(
                    f"\n{_role:10s}: [{str(_audio_id)[:10]}] "
                    + f"{_audio_transcript if _audio_transcript else ''}"
                )
                if _audio_data is not None:
                    print(f"\n{_role:10s}: {_audio_data[:100] + '...'}")

            if tool_calls is not None:
                for _tool_call in tool_calls:
                    if _tool_call["type"] == "function":
                        _function_name = _tool_call["function"]["name"]
                        _function_arguments = _tool_call["function"]["arguments"]
                        print(
                            f"\n{_role:10s}: [{_tool_call['id'][:10]}] "
                            + f"{_function_name}({_function_arguments})"
                        )
                    else:
                        print(f"\n{_role:10s}: {_tool_call}")

        return None
