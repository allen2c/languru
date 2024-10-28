from textwrap import dedent
from typing import Any, Callable, ClassVar, Final, Text

from openai.types.beta.function_tool import FunctionTool
from pydantic import BaseModel, Field

FIELD_FUNCTION_NAME: Final[Text] = "FUNCTION_NAME"
FIELD_FUNCTION_DESCRIPTION: Final[Text] = "FUNCTION_DESCRIPTION"
FIELD_FUNCTION: Final[Text] = "FUNCTION"
FIELD_FUNCTION_ERROR_CONTENT: Final[Text] = "FUNCTION_ERROR_CONTENT"


class FunctionToolRequestBaseModel(BaseModel):
    FUNCTION_NAME: ClassVar[Text]
    FUNCTION_DESCRIPTION: ClassVar[Text]
    FUNCTION: ClassVar[Callable]
    FUNCTION_ERROR_CONTENT: ClassVar[Text] = Field(
        default=dedent(
            """
            The service is currently unavailable. Please try again later.
            """
        ).strip(),
    )

    @classmethod
    def to_function_tool(cls) -> FunctionTool:
        from languru.function_tools.utils import func_tool_from_base_model

        return func_tool_from_base_model(cls)

    @classmethod
    def parse_response_as_tool_content(cls, response: Any) -> Text:
        raise NotImplementedError
