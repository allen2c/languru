from typing import TYPE_CHECKING, List, Sequence, Type

from openai.types.beta.function_tool import FunctionTool
from openai.types.shared.function_definition import FunctionDefinition
from pydantic import BaseModel

if TYPE_CHECKING:
    from languru.function_tools.function_base_model import FunctionToolRequestBaseModel


def func_def_from_base_model(
    base_model_type: (
        Type["FunctionToolRequestBaseModel"]
        | "FunctionToolRequestBaseModel"
        | Type[BaseModel]
        | BaseModel
    ),
) -> "FunctionDefinition":
    from languru.function_tools.function_base_model import (
        FIELD_FUNCTION_DESCRIPTION,
        FIELD_FUNCTION_NAME,
    )

    func_name = getattr(base_model_type, FIELD_FUNCTION_NAME, None)
    if not func_name:
        raise ValueError(
            "The class variable `function_name` is not set for the base model: "
            + f"{base_model_type}"
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
