import logging
import typing

import agents
import httpx
import openai
from openai._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from openai.types.chat import completion_create_params
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_prediction_content_param import (
    ChatCompletionPredictionContentParam,
)
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.shared.chat_model import ChatModel
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.metadata import Metadata
from openai_shared.tools import function_tool_to_chatcmpl_tool_param

logger = logging.getLogger(__name__)


class OpenAIChatCompletionStreamHandler:
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
        self.__stream_options = stream_options
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

    async def run_until_done(self) -> None:
        stream = await self.__openai_client.chat.completions.create(
            messages=self.__messages,
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
            stream_options=self.__stream_options,
            temperature=self.__temperature,
            tool_choice=self.__tool_choice,  # type: ignore
            tools=(
                [function_tool_to_chatcmpl_tool_param(tool) for tool in self.__tools]
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
        async for chunk in stream:
            await self.__on_event(chunk)

    async def __on_event(self, event: ChatCompletionChunk) -> None:
        await self.on_event(event)

    async def on_event(self, event: ChatCompletionChunk) -> None:
        pass
