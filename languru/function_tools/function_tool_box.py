from typing import TYPE_CHECKING, Dict, Iterable, List, Text, Type

import httpx
import openai

from languru.config import console, logger

if TYPE_CHECKING:
    from openai.types.beta.threads import run_submit_tool_outputs_params
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
            if self._debug:
                _debug_print_tool_call_details(tool)

            if tool.function.name in self._func_tool_models:
                func_tool_model = self._func_tool_models[tool.function.name]
                func_request = func_tool_model.from_args_str(tool.function.arguments)

                func_response = func_tool_model.FUNCTION(func_request)
                func_tool_output = (
                    func_tool_model.parse_response_as_assistant_tool_output(
                        func_response, tool_call_id=tool.id
                    )
                )

                if self._debug:
                    _debug_print_tool_output(func_tool_output)
                tool_outputs.append(func_tool_output)

            else:
                logger.error(
                    f"Unknown tool: '{tool.function.name}', "
                    + f"available tools: {list(self._func_tool_models.keys())}"
                )

        return tool_outputs

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
