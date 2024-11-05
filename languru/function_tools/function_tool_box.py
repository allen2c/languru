import json
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Text, Type, Union

import httpx
import openai
from openai.types.beta.function_tool import FunctionTool
from openai.types.beta.function_tool_param import FunctionToolParam
from openai.types.beta.threads import run_submit_tool_outputs_params
from pydantic import BaseModel

from languru.config import console, logger
from languru.utils.openai_utils import rand_openai_id

if TYPE_CHECKING:
    from openai.types.beta.threads.required_action_function_tool_call import (
        RequiredActionFunctionToolCall,
    )
    from openai.types.beta.threads.run import Run

    from languru.function_tools.function_base_model import FunctionToolRequestBaseModel


class FunctionToolBox:
    def __init__(
        self,
        function_tool_models: Iterable[Type["FunctionToolRequestBaseModel"]],
        *,
        debug: bool = False,
    ):
        self._func_tool_models: Dict[Text, Type["FunctionToolRequestBaseModel"]] = {
            t.FUNCTION_NAME: t for t in function_tool_models
        }
        self._debug = debug

    @property
    def function_tools(self) -> List["FunctionTool"]:
        return [t.to_function_tool() for t in self._func_tool_models.values()]

    @property
    def function_tool_params(self) -> List["FunctionToolParam"]:
        return [t.to_function_tool_param() for t in self._func_tool_models.values()]

    def handle_openai_thread_run_tool_calls(
        self, run: "Run", *, openai_client: "openai.OpenAI"
    ) -> List["run_submit_tool_outputs_params.ToolOutput"]:
        tool_outputs = []
        if run.required_action is None:
            return tool_outputs

        # Debug print required tool calls
        if self._debug:
            _debug_print_all_tool_calls(run)

        for tool in run.required_action.submit_tool_outputs.tool_calls:
            tool_outputs.append(self.execute_required_action_function_tool_call(tool))

        return tool_outputs

    def execute_required_action_function_tool_call(
        self, tool: "RequiredActionFunctionToolCall", **kwargs
    ) -> "run_submit_tool_outputs_params.ToolOutput":
        if self._debug:
            _debug_print_tool_call_details(tool)

        try:
            func_tool_output = self.execute_function(
                tool.function.name, tool.function.arguments, tool_call_id=tool.id
            )
            if self._debug:
                _debug_print_tool_output(func_tool_output)
            return func_tool_output

        except openai.NotFoundError:
            logger.error(
                f"Unknown tool: '{tool.function.name}', "
                + f"available tools: {list(self._func_tool_models.keys())}"
            )
            return run_submit_tool_outputs_params.ToolOutput(
                tool_call_id=tool.id,
                output=f"Tool '{tool.function.name}' not available.",
            )

    def execute_function(
        self,
        func_name: Text,
        func_arguments: Union[Text, BaseModel, Dict],
        *,
        tool_call_id: Optional[Text] = None,
        **kwargs,
    ) -> "run_submit_tool_outputs_params.ToolOutput":
        if func_name not in self._func_tool_models:
            raise openai.NotFoundError(
                f"Function tool model with name '{func_name}' not found.",
                response=httpx.Response(status_code=404),
                body=None,
            )

        call_id = tool_call_id or rand_openai_id("call")
        func_tool_model = self._func_tool_models[func_name]

        try:
            func_request = (
                func_tool_model.from_args_str(func_arguments)
                if isinstance(func_arguments, Text)
                else (
                    func_arguments.model_dump_json()
                    if isinstance(func_arguments, BaseModel)
                    else json.dumps(func_arguments)
                )
            )
            func_response = func_tool_model.FUNCTION(func_request)
            return func_tool_model.parse_response_as_assistant_tool_output(
                func_response, tool_call_id=call_id
            )

        except Exception as e:
            logger.exception(e)
            logger.error(
                f"Error executing function '{func_name}' "
                + f"with arguments: {func_arguments},"
                + f" tool call id: {call_id}, returning error content."
            )
            return run_submit_tool_outputs_params.ToolOutput(
                tool_call_id=call_id,
                output=func_tool_model.FUNCTION_ERROR_CONTENT,
            )

    def get_function_tool_model(
        self, function_name: Text
    ) -> Type["FunctionToolRequestBaseModel"]:
        if function_name not in self._func_tool_models:
            raise openai.NotFoundError(
                f"Function tool model with name '{function_name}' not found.",
                response=httpx.Response(status_code=404),
                body=None,
            )
        return self._func_tool_models[function_name]

    def add_function_tool_model(
        self, function_tool_model: Type["FunctionToolRequestBaseModel"]
    ) -> None:
        self._func_tool_models[function_tool_model.FUNCTION_NAME] = function_tool_model

    def remove_function_tool_model(
        self, function_name: Text, *, raise_if_not_found: bool = False
    ) -> None:
        if function_name not in self._func_tool_models:
            if raise_if_not_found:
                raise openai.NotFoundError(
                    f"Function tool model with name '{function_name}' not found.",
                    response=httpx.Response(status_code=404),
                    body=None,
                )
        self._func_tool_models.pop(function_name, None)


def _debug_print_all_tool_calls(run: "Run") -> None:
    if run.required_action is None:
        return

    _tool_calls_names_parts = []
    for tool in run.required_action.submit_tool_outputs.tool_calls:
        _tool_calls_names_parts.append(f"'{tool.function.name}({tool.id})'")
    _tool_calls_expr = ", ".join(_tool_calls_names_parts)
    console.print(
        f"[bold bright_green]Thread({run.thread_id}) Run({run.id}) Tool Calls:[/] "
        + f"[bright_cyan]{_tool_calls_expr}[/]"
    )


def _debug_print_tool_call_details(tool: "RequiredActionFunctionToolCall") -> None:
    console.print(f"\n[bold green]Tool Call (id={tool.id}) Details:[/]")
    console.print(
        f"[bold bright_green]Tool Name:[/] [bright_cyan]{tool.function.name}[/]"
    )
    console.print(
        "[bold bright_green]Tool Arguments:[/] [bright_cyan]"
        + f"{tool.function.arguments}[/]"
    )


def _debug_print_tool_output(
    tool_output: "run_submit_tool_outputs_params.ToolOutput",
) -> None:
    console.print(
        "[bold bright_green]Tool Output:[/] [bright_cyan]"
        + f"{tool_output.get('output') or 'N/A'}[/]"
    )
