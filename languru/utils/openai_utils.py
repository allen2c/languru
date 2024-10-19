import base64
import hashlib
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Text,
    Union,
)
from xml.sax.saxutils import escape as xml_escape

import numpy as np
from numpy.typing import DTypeLike
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from pyassorted.string.rand import rand_str

from languru.config import logger
from languru.types.chat.completions import Message

if TYPE_CHECKING:
    from diskcache import Cache


def rand_openai_id(
    type: Literal[
        "chat_completion",
        "chatcmpl",
        "assistant",
        "asst",
        "thread",
        "message",
        "msg",
        "run",
    ]
) -> Text:
    if type in ("chat_completion", "chatcmpl"):
        return rand_chat_completion_id()
    elif type in ("assistant", "asst"):
        return rand_assistant_id()
    elif type == "thread":
        return rand_thread_id()
    elif type in ("message", "msg"):
        return rand_message_id()
    elif type == "run":
        return rand_run_id()
    else:
        raise ValueError(f"Invalid type: {type}")


def rand_chat_completion_id() -> Text:
    return f"chatcmpl-{rand_str(29)}"


def rand_assistant_id() -> Text:
    return f"asst_{rand_str(24)}"


def rand_thread_id() -> Text:
    return f"thread_{rand_str(24)}"


def rand_message_id() -> Text:
    return f"msg_{rand_str(24)}"


def rand_run_id() -> Text:
    return f"run_{rand_str(24)}"


def ensure_chat_completion_message_params(
    messages: (
        Sequence[ChatCompletionMessageParam]
        | Sequence[Dict[Text, Any]]
        | ChatCompletionMessageParam
        | Dict[Text, Any]
    )
) -> List[ChatCompletionMessageParam]:
    """Ensure the messages parameter is a list of ChatCompletionMessageParam objects."""

    if not messages:
        raise ValueError("The messages parameter is required.")

    return list([messages] if isinstance(messages, Dict) else messages)  # type: ignore


def ensure_openai_chat_completion_message_params(
    messages: Union[
        Sequence["Message"],
        Sequence[Dict[Text, Any]],
        Sequence[ChatCompletionMessageParam],
    ]
) -> List[ChatCompletionMessageParam]:
    """Ensure that the chat completion messages are in the correct format."""

    _messages: List[ChatCompletionMessageParam] = []
    for m in messages:
        if isinstance(m, Message):
            m_dict = m.model_dump()
            if m_dict.get("role") not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role: {m_dict.get('role')}")
            _messages.append(m_dict)  # type: ignore
        elif isinstance(m, Dict):
            if m.get("role") not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role: {m.get('role')}")
            _messages.append(m)  # type: ignore
        elif isinstance(m, ChatCompletionMessageParam.__args__):
            _messages.append(m)
        else:
            raise ValueError(f"Invalid message type: {m}")
    return _messages


def ensure_openai_chat_completion_content(chat_completion: "ChatCompletion") -> Text:
    """Ensure that the chat completion response content is returned."""

    chat_answer = chat_completion.choices[0].message.content
    if chat_answer is None:
        raise ValueError("Failed to get a response content from the OpenAI API.")
    return chat_answer


def messages_to_md5(messages: List[ChatCompletionMessageParam]) -> Text:
    """Convert messages to an MD5 hash."""

    return hashlib.md5(
        json.dumps(messages, sort_keys=True, default=str).encode()
    ).hexdigest()


def messages_to_xml(
    messages: List[ChatCompletionMessageParam] | List[Dict],
    *,
    wrapper_tag: Text = "chat_records",
    indent: Text = "",
) -> Text:
    """Convert a list of chat messages to an XML string.

    This function takes a list of chat messages and converts them into an XML format.
    Each message is represented as a child element under a specified wrapper tag.

    Parameters
    ----------
    messages : List[ChatCompletionMessageParam]
        A list of messages to be converted to XML. Each message should contain a 'role'
        and 'content' field.

    wrapper_tag : Text, optional
        The tag name for the root element of the XML. Default is "chat_records".

    indent : Text, optional
        A string used for indentation in the XML output. Default is an empty string.

    Returns
    -------
    Text
        A string representation of the XML containing the chat messages.
    """

    import xml.etree.ElementTree as ET

    from languru.utils._xml import pretty_xml

    root = ET.Element(xml_escape(wrapper_tag))
    for m in messages:
        _role = m["role"]
        _content: Union[Text, None] = None
        if m.get("content"):
            if isinstance(m.get("content"), Text):
                _content = m["content"]  # type: ignore
            else:
                for _part in m["content"]:  # type: ignore
                    assert isinstance(_part, Dict)
                    if _part.get("text"):
                        _content = _part["text"]  # type: ignore
                    elif _part.get("refusal"):
                        _content = _part["refusal"]  # type: ignore

        if _content is not None:
            child = ET.SubElement(root, _role)
            child.text = f"\n{xml_escape(str(_content)).strip()}\n"

    return pretty_xml(root, indent=indent)


def emb_to_base64(emb: List[float], dtype: DTypeLike = np.float32) -> Text:
    return base64.b64encode(np.array(emb, dtype=dtype).tobytes()).decode("utf-8")


def emb_from_base64(base64_str: Text, dtype: DTypeLike = np.float32) -> List[float]:
    return np.frombuffer(  # type: ignore[no-untyped-call]
        base64.b64decode(base64_str), dtype=dtype
    ).tolist()


def embeddings_create_with_cache(
    *,
    input: Text | Sequence[Text],
    model: Text,
    dimensions: int,
    openai_client: "OpenAI",
    cache: Optional["Cache"],
) -> List[List[float]]:
    if not input:
        return []

    _input = [input] if isinstance(input, Text) else list(input)
    _output: List[Optional[List[float]]] = [None] * len(_input)

    # Check cache existence
    _cached_idx: List[int] = []
    _uncached_idx: List[int] = []
    if cache is not None:
        for i, _inp in enumerate(_input):
            _cached_emb_base64: Optional[Text] = cache.get(_inp)  # type: ignore
            if _cached_emb_base64 is not None:
                logger.debug(f"Embedding cache hit for '{_inp[:24]}...'")
                _output[i] = emb_from_base64(_cached_emb_base64)
                _cached_idx.append(i)
            else:
                _uncached_idx.append(i)
    else:
        _uncached_idx = list(range(len(_input)))

    # Get embeddings from OpenAI
    if _uncached_idx:
        _emb_res = openai_client.embeddings.create(
            input=[_input[i] for i in _uncached_idx],
            model=model,
            dimensions=dimensions,
        )
        for i, emb in zip(_uncached_idx, _emb_res.data):
            if cache is not None:
                logger.debug(f"Caching embedding for '{_input[i][:24]}...'")
                cache.set(_input[i], emb_to_base64(emb.embedding))
            _output[i] = emb.embedding

    # Check if any embeddings failed to be retrieved
    if any(e is None for e in _output):
        raise ValueError("Failed to get embeddings from the OpenAI API.")
    return _output  # type: ignore


def ensure_vector(
    query: Text | List[float],
    *,
    openai_client: Optional["OpenAI"] = None,
    cache: Optional["Cache"],
    input_func: Callable[[Text], List[Text]] = lambda x: [x.strip()],
    embedding_model: Optional[Text] = None,
    embedding_dimensions: Optional[int] = None,
) -> List[float]:
    if isinstance(query, Text):
        query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if isinstance(query, Text):
        if not openai_client:
            raise ValueError(
                "Argument `openai_client` is required to create embeddings."
            )
        if not embedding_model:
            raise ValueError(
                "Argument `embedding_model` is required to create embeddings."
            )
        if not embedding_dimensions:
            raise ValueError(
                "Argument `embedding_dimensions` is required to create embeddings."
            )
        _inputs = input_func(query)
        _vectors = embeddings_create_with_cache(
            input=_inputs,
            model=embedding_model,
            dimensions=embedding_dimensions,
            openai_client=openai_client,
            cache=cache,
        )
        if len(_vectors) != len(_inputs):
            raise ValueError(
                f"Expected {len(_inputs)} vectors, but got {len(_vectors)} vectors."
            )
        _vector = _vectors[0]

    else:
        _vector = query

    if embedding_dimensions is not None:
        if len(_vector) != embedding_dimensions:
            raise ValueError(
                f"Expected vector of length {embedding_dimensions}, "
                + f"but got {len(_vector)}."
            )
    return _vector
