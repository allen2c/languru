from typing import Mapping, Optional, Text, Union

import httpx
import openai.resources
import openai.resources.beta.threads
import openai.resources.beta.threads.runs
from httpx import Timeout
from openai import DEFAULT_MAX_RETRIES, NOT_GIVEN, NotGiven, OpenAI
from openai._compat import cached_property


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

        self.beta = BetaWithToolbox(self)


class BetaWithToolbox(openai.resources.Beta):
    @cached_property
    def threads(self) -> "ThreadsWithToolbox":
        return ThreadsWithToolbox(self._client)


class ThreadsWithToolbox(openai.resources.beta.threads.Threads):
    @cached_property
    def runs(self) -> "RunsWithToolbox":
        return RunsWithToolbox(self._client)


class RunsWithToolbox(openai.resources.beta.threads.runs.Runs):
    pass


if __name__ == "__main__":
    client = OpenAIWithToolbox()
    run = client.beta.threads.runs.create_and_poll(
        assistant_id="asst_123", thread_id="thread_123"
    )
    print(run)
