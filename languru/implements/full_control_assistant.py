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


class FullControlOpenAI(OpenAI):

    beta: "FullControlBeta"

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
        self.beta = FullControlBeta(self)

        # Custom function toolbox
        self.function_toolbox = function_toolbox


class FullControlBeta(openai.resources.Beta):

    _client: FullControlOpenAI

    # Custom threads
    @cached_property
    def threads(self) -> "FullControlThreads":
        return FullControlThreads(self._client)


class FullControlThreads(openai.resources.beta.threads.Threads):

    _client: FullControlOpenAI

    # Custom runs
    @cached_property
    def runs(self) -> "FullControlRuns":
        return FullControlRuns(self._client)


class FullControlRuns(openai.resources.beta.threads.runs.Runs):

    _client: FullControlOpenAI
