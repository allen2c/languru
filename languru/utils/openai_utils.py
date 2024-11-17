import base64
import hashlib
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Text,
    Union,
)
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import rich.box
import rich.panel
from diskcache import Cache
from numpy.typing import DTypeLike
from openai import NotFoundError, OpenAI
from openai.types.beta.assistant import Assistant
from openai.types.beta.function_tool import FunctionTool
from openai.types.beta.function_tool_param import FunctionToolParam
from openai.types.beta.threads.required_action_function_tool_call import (
    RequiredActionFunctionToolCall,
)
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from pyassorted.string.rand import rand_str
from rich.text import Text as RT

from languru.config import console, logger
from languru.types.chat.completions import Message

if TYPE_CHECKING:
    from redis import Redis


style_key_val = (
    lambda k, v, value_new_line=False, end_new_line=True: RT(
        f"{k}", style="bold bright_cyan"
    )
    + RT(":\n" if value_new_line else ": ", style="default")
    + RT(f"{v}", style="bright_blue")
    + RT("\n" if end_new_line else "", style="default")
)

cache = Cache("/tmp/.languru_cache")


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
        "call",
        "tool_call",
        "toolcall",
        "tool",
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
    elif type in ("call", "tool_call", "toolcall", "tool"):
        return rand_tool_call_id()
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


def rand_tool_call_id() -> Text:
    return f"call_{rand_str(24)}"


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


def get_assistant_by_id_or_name(
    id_or_name: Text,
    *,
    openai_client: "OpenAI",
    cache: Optional[Union["Cache", "Redis"]] = None,
    expire: int = 5 * 60,  # 5 minutes
) -> Optional["Assistant"]:
    # Check cache
    if cache is not None:
        _cache_key = f"openai:assistant:{id_or_name}"
        _cached_asst_json: Optional[Text] = cache.get(_cache_key)  # type: ignore
        if _cached_asst_json is not None:
            return Assistant.model_validate_json(_cached_asst_json)

    # Retrieve an existing assistant by ID
    if id_or_name.startswith("asst_"):
        try:
            assistant = openai_client.beta.assistants.retrieve(id_or_name)
        except NotFoundError:
            # Try to find an assistant by name
            pass

    # Search for the assistant by name
    after: Optional[Text] = None
    has_more: bool = True
    assistant: Optional["Assistant"] = None
    while has_more:
        _params: Dict = {k: v for k, v in {"after": after}.items() if v is not None}
        assistants_page = openai_client.beta.assistants.list(**_params)
        if len(assistants_page.data) == 0:
            break
        for _asst in assistants_page.data:
            if _asst.id == id_or_name or _asst.name == id_or_name:
                assistant = _asst
                break
        after = assistants_page.data[-1].id
        has_more = (
            assistants_page.has_more  # type: ignore
            if hasattr(assistants_page, "has_more")
            else False
        )

    # Cache the assistant
    if cache is not None and assistant is not None:
        cache.set(_cache_key, assistant.model_dump_json(), expire)
    return assistant


def ensure_assistant(
    id_or_name: Text,
    *,
    openai_client: "OpenAI",
    assistant_name: Optional[Text] = None,
    assistant_instructions: Optional[Text] = None,
    assistant_model: Text = "gpt-4o-mini",
    assistant_temperature: float = 0.3,
    assistant_tools: Optional[Iterable["FunctionTool"]] = None,
    cache: Optional[Union["Cache", "Redis"]] = cache,
    expire: int = 5 * 60,  # 5 minutes
) -> "Assistant":
    assistant: Optional["Assistant"] = None
    tools: List[FunctionToolParam] = [
        t.model_dump() for t in assistant_tools or []  # type: ignore
    ]

    # Retrieve an existing assistant by ID or name
    assistant = get_assistant_by_id_or_name(
        id_or_name, openai_client=openai_client, cache=cache, expire=expire
    )

    # Create a new assistant
    if assistant is None:
        if assistant_name is None or assistant_instructions is None:
            raise ValueError(
                "Try to create a new assistant, but argument `assistant_name` and "
                + "`assistant_instructions` are not provided."
            )
        logger.info(f"Creating assistant: {assistant_name}")
        assistant = openai_client.beta.assistants.create(
            name=assistant_name,
            instructions=assistant_instructions,
            model=assistant_model,
            temperature=assistant_temperature,
            tools=tools,
        )

    # Update an existing assistant
    else:
        logger.info(f"Updating assistant: {assistant_name}")
        assistant = openai_client.beta.assistants.update(
            assistant_id=assistant.id,
            instructions=assistant_instructions,
            temperature=assistant_temperature,
            tools=tools,
        )

    return assistant


def display_assistant(
    assistant: "Assistant", *, is_print: bool = True, max_length: int = 300
) -> RT:
    if not assistant or not isinstance(assistant, Assistant):
        return RT("Input is not an instance of Assistant.")

    try:
        title = style_key_val("Assistant", assistant.id)
        _tools_names = [
            f"{t.function.name}" for t in assistant.tools if t.type == "function"
        ]
        _tools_expr = ", ".join(_tools_names) or "N/A"
        _inst = (
            (
                assistant.instructions[:max_length] + "..."
                if len(assistant.instructions) > max_length
                else assistant.instructions
            )
            if assistant.instructions
            else "N/A"
        )

        content = RT("")
        content += style_key_val("Name", assistant.name)
        content += style_key_val("Model", assistant.model)
        content += style_key_val("Temperature", assistant.temperature)
        content += style_key_val("Tools", _tools_expr)
        content += style_key_val(
            "Instructions", _inst, value_new_line=True, end_new_line=False
        )

        if is_print:
            console.print(
                rich.panel.Panel(content, title=title, box=rich.box.HORIZONTALS)
            )
    except Exception as e:
        console.print_exception()
        return RT(f"Error displaying assistant: {e}")

    return content


def display_tool_call_details(
    tool: "RequiredActionFunctionToolCall", *, is_print: bool = True
) -> RT:
    if not tool:
        return RT("Input is not an instance of RequiredActionFunctionToolCall.")

    try:
        content = RT("")
        title = style_key_val("Tool Call", tool.id)
        content += style_key_val("Tool Name", tool.function.name)
        content += style_key_val(
            "Tool Arguments", tool.function.arguments, end_new_line=False
        )
        if is_print:
            console.print(
                rich.panel.Panel(content, title=title, box=rich.box.HORIZONTALS)
            )
    except Exception as e:
        logger.exception(e)
        return RT(f"Error displaying tool call details: {e}")
    return content


def display_tool_output(
    tool_output: "ToolOutput", *, is_print: bool = True, max_length: int = 300
) -> RT:
    if not tool_output:
        return RT("Input is not an instance of ToolOutput.")

    try:
        title = style_key_val("Tool Output", tool_output.get("tool_call_id"))
        content = RT("")
        output_content = tool_output.get("output") or "N/A"
        output_content = (
            output_content[:max_length] + "..."
            if len(output_content) > max_length
            else output_content
        )
        content += style_key_val(
            "Output", output_content, value_new_line=True, end_new_line=False
        )

        if is_print:
            console.print(
                rich.panel.Panel(content, title=title, box=rich.box.HORIZONTALS)
            )

    except Exception as e:
        logger.exception(e)
        return RT(f"Error displaying tool output: {e}")

    return content
