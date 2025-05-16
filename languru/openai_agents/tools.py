import typing

import agents
import pydantic


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
