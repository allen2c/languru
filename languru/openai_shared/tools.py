import typing

import agents
import pydantic
from openai.types.responses.function_tool_param import FunctionToolParam


def base_model_to_function_tool(
    base_model: typing.Type[pydantic.BaseModel],
    *,
    name: typing.Text,
    description: typing.Text,
    on_invoke_tool: typing.Callable[
        [agents.RunContextWrapper[typing.Any], str], typing.Awaitable[typing.Any]
    ],
) -> agents.FunctionTool:
    return agents.FunctionTool(
        name=name,
        description=description,
        params_json_schema=validate_json_schema(base_model.model_json_schema()),
        on_invoke_tool=on_invoke_tool,
    )


def function_tool_to_responses_tool_param(
    function_tool: agents.FunctionTool,
) -> FunctionToolParam:
    return FunctionToolParam(
        name=function_tool.name,
        parameters=function_tool.params_json_schema,
        strict=function_tool.strict_json_schema,
        type="function",
        description=function_tool.description,
    )


def validate_json_schema(json_schema: typing.Dict) -> typing.Dict:
    if "required" not in json_schema:
        json_schema["required"] = []

    if "properties" in json_schema:
        for prop_key, prop_value in json_schema["properties"].items():
            # Remove default value from the property, because OpenAI does not accept it
            if "default" in prop_value:
                prop_value.pop("default")

            json_schema["required"].append(prop_key)

    return json_schema
