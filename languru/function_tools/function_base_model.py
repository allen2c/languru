from typing import Any, Callable, ClassVar, Final, List, Sequence, Text, Type

from openai.types.beta.function_tool import FunctionTool
from openai.types.shared.function_definition import FunctionDefinition
from pydantic import BaseModel

FIELD_FUNCTION_NAME: Final[Text] = "FUNCTION_NAME"
FIELD_FUNCTION_DESCRIPTION: Final[Text] = "FUNCTION_DESCRIPTION"
FIELD_FUNCTION: Final[Text] = "FUNCTION"


def func_def_from_base_model(
    base_model_type: (
        Type["FunctionToolRequestBaseModel"]
        | "FunctionToolRequestBaseModel"
        | Type[BaseModel]
        | BaseModel
    ),
) -> "FunctionDefinition":
    func_name = getattr(base_model_type, FIELD_FUNCTION_NAME, None)
    if not func_name:
        raise ValueError(
            "The class variable `function_name` is not set for the base model"
        )
    func_description = getattr(base_model_type, FIELD_FUNCTION_DESCRIPTION, None)
    model_json_schema = base_model_type.model_json_schema()
    model_json_schema.pop("title", None)
    return FunctionDefinition.model_validate(
        {
            "name": func_name,
            "description": func_description,
            "parameters": model_json_schema,
        }
    )


def func_tool_from_base_model(
    base_model_type: (
        Type["FunctionToolRequestBaseModel"]
        | "FunctionToolRequestBaseModel"
        | Type[BaseModel]
        | BaseModel
    ),
) -> "FunctionTool":
    return FunctionTool.model_validate(
        {
            "function": func_def_from_base_model(base_model_type),
            "type": "function",
        }
    )


def func_tools_from_base_models(
    base_model_types: Sequence[
        Type["FunctionToolRequestBaseModel"]
        | "FunctionToolRequestBaseModel"
        | Type[BaseModel]
        | BaseModel
    ],
) -> List["FunctionTool"]:
    return [
        func_tool_from_base_model(base_model_type)
        for base_model_type in base_model_types
    ]


class FunctionToolRequestBaseModel(BaseModel):
    FUNCTION_NAME: ClassVar[Text]
    FUNCTION_DESCRIPTION: ClassVar[Text]
    FUNCTION: ClassVar[Callable]

    @classmethod
    def to_function_tool(cls) -> FunctionTool:
        return func_tool_from_base_model(cls)

    def parse_response_as_tool_content(self, response: Any) -> Text:
        raise NotImplementedError
