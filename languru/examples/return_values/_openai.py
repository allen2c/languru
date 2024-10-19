import random
import time

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.create_embedding_response import CreateEmbeddingResponse

from languru.types.completions import Completion
from languru.types.model import Model

return_chat_completion = ChatCompletion.model_validate(
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxxxxx",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello! How can I assist you today?",
                    "role": "assistant",
                },
            }
        ],
        "created": 1710559936,
        "model": "gpt-3.5-turbo-0125",
        "object": "chat.completion",
        "system_fingerprint": "fp_xxxxxxxx",
        "usage": {
            "completion_tokens": 9,
            "prompt_tokens": 19,
            "total_tokens": 28,
        },
    }
)
return_chat_completion_chunks = [
    ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-xxxx",
            "choices": [{"delta": {"content": "", "role": "assistant"}, "index": 0}],
            "created": 1710565261,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_xxxx",
        }
    ),
    ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-xxxx",
            "choices": [{"delta": {"content": "Hello"}, "index": 0}],
            "created": 1710565261,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_xxxx",
        }
    ),
    ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-xxxx",
            "choices": [{"delta": {"content": "!"}, "index": 0}],
            "created": 1710565261,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_xxxx",
        }
    ),
    ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-xxxx",
            "choices": [{"delta": {"content": " How"}, "index": 0}],
            "created": 1710565261,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_xxxx",
        }
    ),
    ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-xxxx",
            "choices": [{"delta": {"content": " can"}, "index": 0}],
            "created": 1710565261,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_xxxx",
        }
    ),
    ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-xxxx",
            "choices": [{"delta": {"content": " I"}, "index": 0}],
            "created": 1710565261,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_xxxx",
        }
    ),
    ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-xxxx",
            "choices": [{"delta": {"content": " assist"}, "index": 0}],
            "created": 1710565261,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_xxxx",
        }
    ),
    ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-xxxx",
            "choices": [{"delta": {"content": " you"}, "index": 0}],
            "created": 1710565261,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_xxxx",
        }
    ),
    ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-xxxx",
            "choices": [{"delta": {"content": " today"}, "index": 0}],
            "created": 1710565261,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_xxxx",
        }
    ),
    ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-xxxx",
            "choices": [{"delta": {"content": "?"}, "index": 0}],
            "created": 1710565261,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_xxxx",
        }
    ),
    ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-xxxx",
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "created": 1710565261,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_xxxx",
        }
    ),
]
return_text_completion = Completion.model_validate(
    {
        "id": "cmpl-xxxxxxxxxxxxxxxxxxxx",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "text": "\n\nThis is a test.",
            }
        ],
        "created": 1710579700,
        "model": "gpt-3.5-turbo-instruct",
        "object": "text_completion",
        "usage": {"completion_tokens": 6, "prompt_tokens": 5, "total_tokens": 11},
    }
)
return_text_completion_stream = [
    Completion.model_validate(
        {
            "id": "cmpl-xxxxxxxxx",
            "choices": [
                {"finish_reason": None, "index": 0, "logprobs": None, "text": "\n\n"}
            ],
            "created": 1710580063,
            "model": "gpt-3.5-turbo-instruct",
            "object": "text_completion",
        }
    ),
    Completion.model_validate(
        {
            "id": "cmpl-xxxxxxxxx",
            "choices": [
                {"finish_reason": None, "index": 0, "logprobs": None, "text": "This"}
            ],
            "created": 1710580063,
            "model": "gpt-3.5-turbo-instruct",
            "object": "text_completion",
        }
    ),
    Completion.model_validate(
        {
            "id": "cmpl-xxxxxxxxx",
            "choices": [
                {"finish_reason": None, "index": 0, "logprobs": None, "text": " is"}
            ],
            "created": 1710580063,
            "model": "gpt-3.5-turbo-instruct",
            "object": "text_completion",
        }
    ),
    Completion.model_validate(
        {
            "id": "cmpl-xxxxxxxxx",
            "choices": [
                {"finish_reason": None, "index": 0, "logprobs": None, "text": " a"}
            ],
            "created": 1710580063,
            "model": "gpt-3.5-turbo-instruct",
            "object": "text_completion",
        }
    ),
    Completion.model_validate(
        {
            "id": "cmpl-xxxxxxxxx",
            "choices": [
                {"finish_reason": None, "index": 0, "logprobs": None, "text": " test"}
            ],
            "created": 1710580063,
            "model": "gpt-3.5-turbo-instruct",
            "object": "text_completion",
        }
    ),
    Completion.model_validate(
        {
            "id": "cmpl-xxxxxxxxx",
            "choices": [
                {"finish_reason": None, "index": 0, "logprobs": None, "text": "."}
            ],
            "created": 1710580063,
            "model": "gpt-3.5-turbo-instruct",
            "object": "text_completion",
        }
    ),
    Completion.model_validate(
        {
            "id": "cmpl-xxxxxxxxx",
            "choices": [
                {"finish_reason": "stop", "index": 0, "logprobs": None, "text": ""}
            ],
            "created": 1710580063,
            "model": "gpt-3.5-turbo-instruct",
            "object": "text_completion",
        }
    ),
]
return_embedding = CreateEmbeddingResponse.model_validate(
    {
        "data": [
            {
                "embedding": [random.uniform(-1, 1) for _ in range(1536)],
                "index": 0,
                "object": "embedding",
            },
            {
                "embedding": [random.uniform(-1, 1) for _ in range(1536)],
                "index": 1,
                "object": "embedding",
            },
        ],
        "model": "text-embedding-ada-002",
        "object": "list",
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }
)
return_model = Model.model_validate(
    {
        "id": "model_id",
        "created": int(time.time()),
        "object": "model",
        "owned_by": "test",
    }
)
return_moderation_create = {
    "id": "modr-AJsgNe8mxIm471iwKXh74Q7PiajVl",
    "model": "text-moderation-007",
    "results": [
        {
            "categories": {
                "harassment": True,
                "harassment_threatening": True,
                "hate": False,
                "hate_threatening": False,
                "illicit": None,
                "illicit_violent": None,
                "self_harm": False,
                "self_harm_instructions": False,
                "self_harm_intent": False,
                "sexual": False,
                "sexual_minors": False,
                "violence": True,
                "violence_graphic": False,
                "self-harm": False,
                "sexual/minors": False,
                "hate/threatening": False,
                "violence/graphic": False,
                "self-harm/intent": False,
                "self-harm/instructions": False,
                "harassment/threatening": True,
            },
            "category_applied_input_types": None,
            "category_scores": {
                "harassment": 0.5215635299682617,
                "harassment_threatening": 0.5694745779037476,
                "hate": 0.22706663608551025,
                "hate_threatening": 0.023547329008579254,
                "illicit": None,
                "illicit_violent": None,
                "self_harm": 2.227119921371923e-06,
                "self_harm_instructions": 1.1198755256458526e-09,
                "self_harm_intent": 1.646940972932498e-06,
                "sexual": 1.1726012417057063e-05,
                "sexual_minors": 7.107352217872176e-08,
                "violence": 0.9971134662628174,
                "violence_graphic": 3.391829886822961e-05,
                "self-harm": 2.227119921371923e-06,
                "sexual/minors": 7.107352217872176e-08,
                "hate/threatening": 0.023547329008579254,
                "violence/graphic": 3.391829886822961e-05,
                "self-harm/intent": 1.646940972932498e-06,
                "self-harm/instructions": 1.1198755256458526e-09,
                "harassment/threatening": 0.5694745779037476,
            },
            "flagged": True,
        }
    ],
}
