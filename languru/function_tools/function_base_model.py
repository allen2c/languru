import json
from textwrap import dedent
from typing import Any, Callable, ClassVar, Final, Text

from json_repair import repair_json
from openai.types.beta.function_tool import FunctionTool
from openai.types.beta.function_tool_param import FunctionToolParam
from openai.types.beta.threads import run_submit_tool_outputs_params
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel

FIELD_FUNCTION_NAME: Final[Text] = "FUNCTION_NAME"
FIELD_FUNCTION_DESCRIPTION: Final[Text] = "FUNCTION_DESCRIPTION"
FIELD_FUNCTION: Final[Text] = "FUNCTION"
FIELD_FUNCTION_ERROR_CONTENT: Final[Text] = "FUNCTION_ERROR_CONTENT"


class FunctionToolRequestBaseModel(BaseModel):
    FUNCTION_NAME: ClassVar[Text]
    FUNCTION_DESCRIPTION: ClassVar[Text]
    FUNCTION: ClassVar[Callable]
    FUNCTION_ERROR_CONTENT: ClassVar[Text] = dedent(
        """
        The service is currently unavailable. Please try again later.
        """
    ).strip()

    @classmethod
    def to_chat_completion_tool_param(cls) -> "ChatCompletionToolParam":
        from languru.function_tools.utils import func_def_from_base_model

        return ChatCompletionToolParam(
            type="function",
            function=func_def_from_base_model(cls).model_dump(),  # type: ignore
        )

    @classmethod
    def to_function_tool(cls) -> "FunctionTool":
        from languru.function_tools.utils import func_tool_from_base_model

        return func_tool_from_base_model(cls)

    @classmethod
    def to_function_tool_param(cls) -> "FunctionToolParam":
        return cls.to_function_tool().model_dump(exclude_none=True)  # type: ignore

    @classmethod
    def parse_response_as_tool_content(cls, response: Any) -> Text:
        raise NotImplementedError

    @classmethod
    def parse_response_as_openai_tool_message_param(
        cls, response: Any, *, tool_call_id: Text
    ) -> ChatCompletionToolMessageParam:
        return ChatCompletionToolMessageParam(
            content=cls.parse_response_as_tool_content(response),
            role="tool",
            tool_call_id=tool_call_id,
        )

    @classmethod
    def parse_response_as_assistant_tool_output(
        cls, response: Any, *, tool_call_id: Text
    ) -> "run_submit_tool_outputs_params.ToolOutput":
        return run_submit_tool_outputs_params.ToolOutput(
            output=cls.parse_response_as_tool_content(response),
            tool_call_id=tool_call_id,
        )

    @classmethod
    def from_args_str(cls, args_str: Text):
        func_kwargs = (
            json.loads(repair_json(args_str)) if args_str else {}  # type: ignore
        )
        return cls.model_validate(func_kwargs)
