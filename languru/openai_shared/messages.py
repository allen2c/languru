import copy as copy_lib
import typing

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


def sanitize_chatcmpl_message_input(
    message: ChatCompletionMessageParam,
) -> ChatCompletionMessageParam:
    # Drop assistant audio data
    if message["role"] == "assistant":
        if _msg_audio := message.get("audio"):
            assert "id" in _msg_audio, "audio.id is required in assistant message"
            for key in list(_msg_audio.keys()):
                if key != "id":
                    _msg_audio.pop(key, None)  # type: ignore

    return message


def sanitize_chatcmpl_messages_input(
    messages: typing.List[ChatCompletionMessageParam],
    *,
    copy: bool = True,
) -> typing.List[ChatCompletionMessageParam]:
    if copy:
        messages = copy_lib.deepcopy(messages)

    messages = [sanitize_chatcmpl_message_input(message) for message in messages]

    return messages
