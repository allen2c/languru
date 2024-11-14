import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Optional,
    Text,
    Tuple,
    Type,
    Union,
)

import httpx
import openai
from openai.types.beta.function_tool import FunctionTool
from openai.types.beta.threads import run_submit_tool_outputs_params
from openai.types.shared.function_definition import FunctionDefinition
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
        max_workers: int = 5,
        debug: bool = False,
    ):
        self._func_tool_models: Dict[Text, Type["FunctionToolRequestBaseModel"]] = {
            t.FUNCTION_NAME: t for t in function_tool_models
        }
        self.max_workers = max_workers
        self.debug = debug

    @property
    def function_tools(self) -> List["FunctionTool"]:
        return [t.to_function_tool() for t in self._func_tool_models.values()]

    def use_tool(
        self,
        name: Text,
        arguments: Union[Text, BaseModel, Dict],
        tool_call_id: Optional[Text] = None,
    ) -> "run_submit_tool_outputs_params.ToolOutput":
        func_tool_output = self.execute_function(
            name, arguments, tool_call_id=tool_call_id
        )
        if self.debug:
            _debug_print_tool_output(func_tool_output)
        return func_tool_output

    def handle_openai_thread_run_tool_calls(
        self,
        run: "Run",
        *,
        openai_client: "openai.OpenAI",
        submit_tool_outputs: bool = True,
    ) -> Tuple[List["run_submit_tool_outputs_params.ToolOutput"], Optional["Run"]]:
        tool_outputs = []
        if run.required_action is None:
            return (tool_outputs, None)

        # Prepare function execution parameters
        execute_func_params: List[Tuple[Text, Text, Optional[Text]]] = []
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            self.raise_if_no_function_tool_model(tool.function.name)
            execute_func_params.append(
                (tool.function.name, tool.function.arguments, tool.id)
            )
            if self.debug:
                _debug_print_tool_call_details(tool)

        # Execute functions in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    partial(
                        self.use_tool,
                        func_name,
                        func_arguments,
                        tool_call_id=tool_call_id,
                    )
                )
                for func_name, func_arguments, tool_call_id in execute_func_params
            ]
            tool_outputs = [future.result() for future in futures]

        # Submit tool outputs if requested
        run_submitted = (
            openai_client.beta.threads.runs.submit_tool_outputs(
                thread_id=run.thread_id, run_id=run.id, tool_outputs=tool_outputs
            )
            if submit_tool_outputs
            else None
        )
        return (tool_outputs, run_submitted)

    def execute_function(
        self,
        func_name: Text,
        func_arguments: Union[Text, BaseModel, Dict],
        *,
        tool_call_id: Optional[Text] = None,
        **kwargs,
    ) -> "run_submit_tool_outputs_params.ToolOutput":
        self.raise_if_no_function_tool_model(func_name)

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
        self.raise_if_no_function_tool_model(function_name)
        return self._func_tool_models[function_name]

    def add_function_tool_model(
        self, function_tool_model: Type["FunctionToolRequestBaseModel"]
    ) -> None:
        self._func_tool_models[function_tool_model.FUNCTION_NAME] = function_tool_model

    def remove_function_tool_model(
        self, function_name: Text, *, raise_if_not_found: bool = False
    ) -> None:
        if raise_if_not_found:
            self.raise_if_no_function_tool_model(function_name)
        self._func_tool_models.pop(function_name, None)

    def has_function_tool_model(self, function_name: Text) -> bool:
        return function_name in self._func_tool_models

    def raise_if_no_function_tool_model(self, function_name: Text) -> None:
        if self.has_function_tool_model(function_name) is False:
            raise openai.NotFoundError(
                f"Function tool model with name '{function_name}' not found.",
                response=httpx.Response(status_code=404),
                body=None,
            )

    def to_definitions(self) -> List[FunctionDefinition]:
        return [t.to_function_definition() for t in self._func_tool_models.values()]


def _debug_print_tool_call_details(tool: "RequiredActionFunctionToolCall") -> None:
    try:
        console.print(f"\n[bold green]Tool Call (id={tool.id}) Details:[/]")
        console.print(
            f"[bold bright_green]Tool Name:[/] [bright_cyan]{tool.function.name}[/]"
        )
        console.print(
            "[bold bright_green]Tool Arguments:[/] [bright_cyan]"
            + f"{tool.function.arguments}[/]"
        )
    except Exception as e:
        logger.exception(e)


def _debug_print_tool_output(
    tool_output: "run_submit_tool_outputs_params.ToolOutput",
) -> None:
    try:
        console.print(
            "[bold bright_green]Tool Output:[/] [bright_cyan]"
            + f"{tool_output.get('output') or 'N/A'}[/]"
        )
    except Exception as e:
        logger.exception(e)
