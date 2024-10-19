import os
import time
from typing import Iterable, List, Literal, Optional, Sequence, Text, Union, cast

import httpx
import openai
import voyageai
from openai import OpenAI
from openai import resources as OpenAIResources
from openai._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from openai.pagination import SyncPage
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.model import Model

from languru.exceptions import CredentialsNotProvided
from languru.openai_plugins.clients.utils import openai_init_parameter_keys
from languru.types.models import MODELS_VOYAGE
from languru.types.rerank import RerankingObject


class VoyageModels(OpenAIResources.Models):

    supported_models = frozenset(MODELS_VOYAGE)

    def retrieve(
        self,
        model: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        **kwargs,
    ) -> "Model":
        if model in self.supported_models:
            return Model.model_validate(
                {
                    "id": model,
                    "created": int(time.time()),
                    "object": "model",
                    "owned_by": "voyage",
                }
            )
        else:
            error_message = (
                f"Model {model} not found. Supported models are {self.supported_models}"
            )
            raise openai.NotFoundError(
                error_message,
                response=httpx.Response(status_code=404, text=error_message),
                body=None,
            )

    def list(
        self,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        **kwargs,
    ) -> "SyncPage[Model]":
        created = int(time.time())
        models = [
            Model.model_validate(
                {
                    "id": model,
                    "created": created,
                    "object": "model",
                    "owned_by": "voyage",
                }
            )
            for model in self.supported_models
        ]
        return SyncPage(data=models, object="list")


class VoyageEmbeddings(OpenAIResources.Embeddings):

    _client: "VoyageOpenAI"

    def create(
        self,
        *,
        input: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]],
        model: Text,
        dimensions: int | NotGiven = NOT_GIVEN,
        encoding_format: Literal["float", "base64"] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> CreateEmbeddingResponse:
        """Create an embedding for the input text or texts."""

        input = [input] if isinstance(input, Text) else input
        input = cast(List[Text], input)

        embed_res = self._client.voyageai_client.embed(texts=input, model=model)

        return CreateEmbeddingResponse.model_validate(
            {
                "data": [
                    {
                        "embedding": emb,
                        "index": idx,
                        "object": "embedding",
                    }
                    for idx, emb in enumerate(embed_res.embeddings)
                ],
                "model": model,
                "object": "list",
                "usage": {
                    "prompt_tokens": embed_res.total_tokens,
                    "total_tokens": embed_res.total_tokens,
                },
            }
        )

    def rerank(
        self,
        query: Text,
        documents: Sequence[Text],
        model: Optional[Text] = None,
        top_k: Optional[int] = None,
        truncation: bool = True,
    ) -> "RerankingObject":
        model = model or self._client.default_rerank_model

        rerank_res = self._client.voyageai_client.rerank(
            query=query,
            documents=list(documents),
            model=model,
            top_k=top_k,
            truncation=truncation,
        )
        return RerankingObject.model_validate(
            {
                "results": [res._asdict() for res in rerank_res.results],
                "total_tokens": rerank_res.total_tokens,
            }
        )


class VoyageOpenAI(OpenAI):
    models: VoyageModels
    embeddings: VoyageEmbeddings

    voyageai_client: voyageai.Client

    def __init__(self, *, api_key: Optional[Text] = None, **kwargs):
        api_key = api_key or os.getenv("VOYAGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise CredentialsNotProvided("Voyage API key is not provided.")
        kwargs["api_key"] = api_key
        kwargs = {k: v for k, v in kwargs.items() if k in openai_init_parameter_keys}

        super().__init__(**kwargs)

        self.models = VoyageModels(self)
        self.embeddings = VoyageEmbeddings(self)

        self.voyageai_client = voyageai.Client(api_key=api_key)
        self.default_rerank_model = "rerank-2-lite"
