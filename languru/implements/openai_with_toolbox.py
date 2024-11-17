from typing import Mapping, Optional, Text, Union

import httpx
import openai._types
import openai._utils
import openai.resources
import openai.resources.beta.threads
import openai.resources.beta.threads.runs
import openai.types.beta.threads.run
from httpx import Timeout
from openai import DEFAULT_MAX_RETRIES, NOT_GIVEN, NotGiven, OpenAI
from openai._compat import cached_property

from languru.function_tools.function_tool_box import FunctionToolBox


class OpenAIWithToolbox(OpenAI):

    beta: "BetaWithToolbox"

    def __init__(
        self,
        *,
        api_key: Optional[Text] = None,
        organization: Optional[Text] = None,
        project: Optional[Text] = None,
        base_url: Optional[Text | httpx.URL] = None,
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Optional[Mapping[Text, Text]] = None,
        default_query: Optional[Mapping[Text, object]] = None,
        http_client: httpx.Client | None = None,
        _strict_response_validation: bool = False,
        function_toolbox: Optional[FunctionToolBox] = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )

        # Custom beta
        self.beta = BetaWithToolbox(self)

        # Custom function toolbox
        self.function_toolbox = function_toolbox


class BetaWithToolbox(openai.resources.Beta):

    _client: OpenAIWithToolbox

    # Custom threads
    @cached_property
    def threads(self) -> "ThreadsWithToolbox":
        return ThreadsWithToolbox(self._client)


class ThreadsWithToolbox(openai.resources.beta.threads.Threads):

    _client: OpenAIWithToolbox

    # Custom runs
    @cached_property
    def runs(self) -> "RunsWithToolbox":
        return RunsWithToolbox(self._client)


class RunsWithToolbox(openai.resources.beta.threads.runs.Runs):

    _client: OpenAIWithToolbox

    def poll(
        self,
        run_id: str,
        thread_id: str,
        extra_headers: openai._types.Headers | None = None,
        extra_query: openai._types.Query | None = None,
        extra_body: openai._types.Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        poll_interval_ms: int | NotGiven = NOT_GIVEN,
    ) -> "openai.types.beta.threads.run.Run":
        """
        A helper to poll a run status until it reaches a terminal state. More
        information on Run lifecycles can be found here:
        https://platform.openai.com/docs/assistants/how-it-works/runs-and-run-steps
        """
        extra_headers = {"X-Stainless-Poll-Helper": "true", **(extra_headers or {})}

        if openai._utils.is_given(poll_interval_ms):
            extra_headers["X-Stainless-Custom-Poll-Interval"] = str(poll_interval_ms)

        terminal_states = {
            "requires_action",
            "cancelled",
            "completed",
            "failed",
            "expired",
            "incomplete",
        }
        while True:
            response = self.with_raw_response.retrieve(
                thread_id=thread_id,
                run_id=run_id,
                extra_headers=extra_headers,
                extra_body=extra_body,
                extra_query=extra_query,
                timeout=timeout,
            )

            run = response.parse()

            # OVERRIDE: Implement function tool calls
            if run.status == "requires_action" and self._client.function_toolbox:
                self._client.function_toolbox.handle_openai_thread_run_tool_calls(
                    run, openai_client=self._client
                )
            # END OVERRIDE

            # Return if we reached a terminal state
            elif run.status in terminal_states:
                return run

            if not openai._utils.is_given(poll_interval_ms):
                from_header = response.headers.get("openai-poll-after-ms")
                if from_header is not None:
                    poll_interval_ms = int(from_header)
                else:
                    poll_interval_ms = 1000

            self._sleep(poll_interval_ms / 1000)


if __name__ == "__main__":
    client = OpenAIWithToolbox()
    run = client.beta.threads.runs.create_and_poll(
        assistant_id="asst_123", thread_id="thread_123"
    )
    print(run)
