import itertools
import json
import string
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    ParamSpec,
    Sequence,
    Text,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import pytz
from openai.types.beta.threads.message import Message as ThreadsMessage
from pydantic import BaseModel
from pydantic_core import ValidationError
from rich import box
from rich import print as rich_print
from rich.style import Style, StyleType
from rich.table import Table
from rich.text import Text as RichText

from languru.config import console, logger
from languru.types.chat.completions import Message

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


def should_str_or_none(value: Text | Any) -> Optional[Text]:
    if isinstance(value, Text):
        return value
    elif value is None:
        return None
    logger.warning(f"Value {value} is not a string, returning None")
    return None


def should_str(value: Text | Any) -> Text:
    if should_str_or_none(value) is None:
        raise ValueError(f"Value {value} is not a string")
    return value


def must_list_or_none(
    value: List[T] | Any, return_none_if_empty: bool = False
) -> Optional[List[T]]:
    if isinstance(value, List):
        if return_none_if_empty and len(value) == 0:
            return None
        return value
    elif isinstance(value, Tuple):
        if return_none_if_empty and len(value) == 0:
            return None
        return list(value)
    elif value is None:
        return None
    else:
        return [value]


def must_list(value: List[T] | Any) -> List[T]:
    to_items = must_list_or_none(value)
    if to_items is None:
        raise ValueError(f"Could not convert {value} to a list")
    return to_items


def debug_print(
    *values: Any,
    title: Text = "Debug Print",
    box: box.Box | None = box.HEAVY_HEAD,
    colors: List[StyleType] = [
        "bright_blue",
        "bright_cyan",
        "bright_green",
        "bright_magenta",
    ],
) -> None:
    if not values:
        return
    tb = Table(title=RichText(title), box=box, show_header=False)
    for idx, value in enumerate(values):
        style = colors[idx % len(colors)]
        if isinstance(value, BaseModel):
            tb.add_row(value.model_dump_json(indent=2), style=style)
        elif isinstance(value, dict):
            tb.add_row(json.dumps(value, indent=2, ensure_ascii=False), style=style)
        else:
            tb.add_row(str(value), style=style)
    rich_print(tb)


def replace_right(source_str: Text, old: Text, new: Text, occurrence: int = -1) -> Text:
    return source_str[::-1].replace(old[::-1], new[::-1], occurrence)[::-1]


def str_strong_casefold(text: Text) -> Text:
    return text.strip().replace("-_. ", "").casefold()


def remove_punctuation(input_string: Text, extra_punctuation: Text = "") -> Text:
    """Remove punctuations from the input string."""

    extended_punctuation = (
        string.punctuation + "，？！（）【】《》“”‘’；：" + extra_punctuation
    )
    translator = str.maketrans("", "", extended_punctuation)
    return input_string.translate(translator)


def ensure_list(value: Any) -> List:
    if isinstance(value, Sequence):
        return list(value)
    if value is None:
        return []
    return [value]


def display_messages(
    messages: Union[
        Sequence["Message"],
        Sequence[Dict[Text, Any]],
        Sequence["ChatCompletionMessageParam"],
        Sequence["ThreadsMessage"],
    ],
    *,
    is_print: bool = True,
    table_title: Text = "Messages",
    table_width: int = 120,
    extra_newline_table_start: bool = True,
    extra_newline_message_end: bool = True,
) -> Text:
    """Display messages in a human-readable format."""

    if not messages:
        raise ValueError("No messages to display.")

    # Convert messages to dictionaries
    _messages = [
        m.model_dump() if isinstance(m, BaseModel) else dict(m) for m in messages
    ]

    # Initialize output
    out = ""
    table: Optional["Table"] = None
    if is_print:
        table = Table(title=table_title, width=table_width)
        table.add_column("Role", justify="right", style="bold cyan")
        table.add_column("Content", justify="left")

    # Read messages
    for m in _messages:
        role = str(m.get("role") or "Unknown").capitalize()
        content = str(m.get("content") or "n/a")
        tool_calls: List[Dict] = m.get("tool_calls") or []  # type: ignore
        if isinstance(content, List):  # OpenAI Threads messages
            _content = ""
            for content_block in content:
                content_block = cast(Dict, content_block)
                if content_block.get("type") == "image_file":
                    _image_file = content_block.get("image_file") or {}
                    _image_id = _image_file.get("file_id") or "n/a"
                    _content += f"<image_file file_id={_image_id}/>"
                elif content_block.get("type") == "image_url":
                    _image_url = content_block.get("image_url") or {}
                    _url = _image_url.get("url") or "n/a"
                    _content += f"<image_url url={_url}/>"
                elif content_block.get("type") == "text":
                    _content_text = content_block.get("text") or {}
                    _content_text_value = _content_text.get("value") or "n/a"
                    _content += str(_content_text_value)
                else:
                    _content += str(content_block)
            content = _content
        elif tool_calls:
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id") or "n/a"
                func_name = get_safe_value(tool_call, "function", "name")
                func_args_str = get_safe_value(tool_call, "function", "arguments")
                content += f"\n\nToolCall:{tool_call_id}:{func_name}({func_args_str})"
        else:
            content = str(content)

        content = content.strip()
        if extra_newline_message_end:
            content += "\n"
        if is_print:
            table = cast(Table, table)
            table.add_row(role.rjust(9), content)
        out += f"\n\n{role.capitalize()}:\n{content}"
        out = out.strip()

    if is_print:
        if extra_newline_table_start:
            console.print("\n")
        console.print(table)
    return out


def named_tuples_to_dicts(named_tuples: Sequence[NamedTuple]) -> List[Dict]:
    """Convert named tuples to dictionaries."""

    return [nt._asdict() for nt in named_tuples]


def json_dumps(data: Any, indent: Optional[Union[int, Text]] = None) -> Text:
    return json.dumps(data, indent=indent, ensure_ascii=False)


def dummy_generator_func(
    generator: Union[
        Generator[T, None, None],
        Iterable[T],
    ],
) -> Callable[[], Generator[T, None, None]]:
    """Create a dummy generator function."""

    def dummy_generator() -> Generator[T, None, None]:
        for item in generator:
            yield item

    return dummy_generator


def display_object(obj: object) -> Text:
    """Display an object in a human-readable format."""

    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


@overload
def model_dump(obj: Sequence) -> List[Dict]: ...


@overload
def model_dump(obj: None) -> None: ...


@overload
def model_dump(obj: Any) -> Dict: ...


def model_dump(obj: Any) -> Optional[Union[Dict, List[Dict]]]:
    """Dump the model in a dictionary format."""

    if obj is None:
        return None
    elif isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, Sequence) and not isinstance(obj, Text):
        return [model_dump(item) for item in obj]
    return json.loads(json.dumps(obj, default=str))


def debug_print_banner(
    content: Text, title: Text = "Title", truncate: int = 500, debug: bool = True
):
    try:
        if debug:
            tag_style = Style(color="green", underline=True, bold=True)
            content = content[:truncate]
            console.print(f"\n<{title}>", style=tag_style)
            console.print(content)
            console.print(f"</{title}>\n", style=tag_style)
    except Exception:
        console.print_exception()


def try_or_none(
    func: Callable,
    *args,
    _print_error: bool = False,
    _error_message: Optional[Text] = None,
    **kwargs,
):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if _print_error:
            console.print_exception()
        if _error_message:
            console.print(_error_message.format(error=e))
        return None


async def try_await_or_none(
    func: Callable[P, Coroutine[None, None, R]],
    *args,
    _print_error: bool = False,
    _error_message: Optional[Text] = None,
    **kwargs,
):
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if _print_error:
            console.print_exception()
        if _error_message:
            console.print(_error_message.format(error=e))
        return None


def choice_first(items: T | Sequence[T]) -> T | None:
    if isinstance(items, Text):
        return items  # type: ignore
    elif isinstance(items, Sequence):
        if len(items) == 0:
            return None
        return items[0]
    else:
        return items


def is_validate_filename(value: Text) -> bool:
    r"""Validate a filename against common restrictions.

    This function checks if the provided filename is valid based on the following criteria:
    - It should not be empty.
    - It should not contain any invalid characters (e.g., <, >, :, ", /, \, |, ?, *).
    - It should not be a reserved name in Windows (e.g., CON, PRN, AUX, NUL, COM1, LPT1, etc.).
    - It should not exceed the common maximum filename length of 255 characters.

    Args:
        value (Text): The filename to validate.

    Raises:
        ValidationError: If the filename is invalid based on the criteria above.

    Returns:
        bool: True if the filename is valid, otherwise raises an exception.
    """  # noqa: E501

    invalid_windows_chars = r'<>:"/\\|?*'
    invalid_unix_chars = r"/"
    invalid_chars = set(invalid_windows_chars + invalid_unix_chars)

    # Windows reserved filenames
    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }

    # Check for empty filename
    if not value:
        raise ValidationError("Filename cannot be empty.")

    # Check for invalid characters
    if any(char in invalid_chars for char in value):
        raise ValidationError(f"Filename contains invalid characters: {invalid_chars}")

    # Check for reserved names (case-insensitive)
    if value.upper().split(".")[0] in reserved_names:
        raise ValidationError("Filename is a reserved name in Windows.")

    # Check for length limits (common max filename length is 255 characters)
    if len(value) > 255:
        raise ValidationError(
            "Filename is too long; it must be 255 characters or fewer."
        )

    return True


def read_jsonl(path: Text, cast: Type[T]) -> List[T]:
    _path = Path(path)
    _path.touch(exist_ok=True)

    records = []
    with open(_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(cast(**json.loads(line)))
    return records


def append_jsonl(data: Dict, *, path: Text):
    _path = Path(path)
    _path.touch(exist_ok=True)
    with open(_path, "a") as f:
        f.write(json.dumps(data) + "\n")


def chunks(
    items: Iterable[T], batch_size: int = 100
) -> Generator[Tuple[T, ...], None, None]:
    """A helper function to break an iterable into chunks of size batch_size."""

    it = iter(items)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def display_datetime_now(tz: pytz.BaseTzInfo = pytz.timezone("Asia/Taipei")) -> Text:

    now = datetime.now(tz)
    return now.strftime("%A, %B %d, %Y %I:%M %p (%Z, UTC%z)")


def get_safe_value(obj: Dict, *keys: Text, default: Any = None) -> Any:
    """Safely get nested dictionary values"""

    try:
        result = obj
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError, IndexError):
        return default
