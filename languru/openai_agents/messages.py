import typing

import agents
from openai.types.responses.easy_input_message_param import (
    EasyInputMessageParam,
)
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput, Message
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_output_message_param import (
    Content,
    ResponseOutputMessageParam,
)
from openai.types.responses.response_reasoning_item_param import (
    ResponseReasoningItemParam,
    Summary,
)

FAKE_ID: typing.Final = "__fake_id__"


class MessageBuilder:
    @staticmethod
    def easy_input_message(
        *,
        content: typing.Union[str, ResponseInputMessageContentListParam],
        role: typing.Literal["user", "assistant", "system", "developer"],
    ) -> agents.TResponseInputItem:
        return EasyInputMessageParam(
            content=content,
            role=role,
            type="message",
        )

    @staticmethod
    def message(
        *,
        content: ResponseInputMessageContentListParam,
        role: typing.Literal["user", "system", "developer"],
    ) -> agents.TResponseInputItem:
        return Message(
            content=content,
            role=role,
            status="in_progress",
            type="message",
        )

    @staticmethod
    def response_output_message(
        *,
        id: typing.Text = FAKE_ID,
        content: typing.Iterable[Content],
        role: typing.Literal["assistant"] = "assistant",
        status: typing.Literal["in_progress", "completed", "incomplete"],
        type: typing.Literal["message"] = "message",
    ) -> agents.TResponseInputItem:
        return ResponseOutputMessageParam(
            id=id,
            content=content,
            role=role,
            status=status,
            type=type,
        )

    @staticmethod
    def response_function_tool_call(
        *,
        arguments: str,
        call_id: str,
        name: str,
        type: typing.Literal["function_call"] = "function_call",
    ) -> agents.TResponseInputItem:
        return ResponseFunctionToolCallParam(
            arguments=arguments,
            call_id=call_id,
            name=name,
            type=type,
        )

    @staticmethod
    def function_call_output(
        *,
        call_id: str,
        output: str,
        type: typing.Literal["function_call_output"] = "function_call_output",
    ) -> agents.TResponseInputItem:
        return FunctionCallOutput(
            call_id=call_id,
            output=output,
            type=type,
        )

    @staticmethod
    def response_reasoning_item(
        *,
        id: typing.Text = FAKE_ID,
        summary: typing.Iterable[Summary],
        type: typing.Literal["reasoning"] = "reasoning",
    ) -> agents.TResponseInputItem:
        return ResponseReasoningItemParam(
            id=id,
            summary=summary,
            type=type,
        )
