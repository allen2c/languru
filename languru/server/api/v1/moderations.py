from logging import Logger
from typing import Optional, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from openai import OpenAI
from openai.types.moderation import (
    Categories,
    CategoryAppliedInputTypes,
    CategoryScores,
)
from openai.types.moderation_create_response import ModerationCreateResponse
from pyassorted.asyncio.executor import run_func
from pydantic import BaseModel

from languru.config import logger as languru_logger
from languru.server.config import ServerBaseSettings
from languru.server.deps.common import app_settings
from languru.server.deps.openai_clients import openai_clients
from languru.server.utils.common import get_value_from_app
from languru.types.moderations import ModerationRequest
from languru.types.organizations import OrganizationType
from languru.utils.common import display_object

router = APIRouter()


def depends_openai_client_moderation_request(
    request: "Request",
    org_type: Optional[OrganizationType] = Depends(openai_clients.depends_org_type),
    moderation_request: ModerationRequest = Body(
        ...,
        openapi_examples={
            "Quick example": {
                "summary": "A quick example of a moderation request",
                "description": "A quick example of a moderation request",
                "value": {"input": "I want to kill them."},
            }
        },
    ),
) -> Tuple[OpenAI, ModerationRequest]:
    logger = get_value_from_app(
        request.app, key="logger", value_typing=Logger, default=languru_logger
    )

    if org_type is None:
        org_type = openai_clients.org_from_model(
            moderation_request.model or ModerationRequest.model_fields["model"].default
        )
    if org_type is None:
        raise HTTPException(status_code=400, detail="Organization type not found.")

    openai_client = openai_clients.org_to_openai_client(org_type)
    if moderation_request.model is not None:
        moderation_request.model = openai_clients.model_strip_org(
            moderation_request.model, org_type
        )
    logger.debug(
        f"Organization type: '{org_type}', "
        + f"openAI client: '{display_object(openai_client)}', "
        + f"model: '{moderation_request.model}'"
    )
    return (openai_client, moderation_request)


class ModerationsHandler:
    async def handle_moderations_request(
        self,
        request: "Request",
        *args,
        moderation_request: "ModerationRequest",
        openai_client: "OpenAI",
        settings: "ServerBaseSettings",
    ) -> "ModerationCreateResponse":
        return await run_func(
            openai_client.moderations.create,
            **moderation_request.model_dump(exclude_none=True),
        )


@router.post("/moderations")
async def request_moderations(
    request: Request,
    openai_client_moderation_request: Tuple[OpenAI, ModerationRequest] = Depends(
        depends_openai_client_moderation_request
    ),
    settings: ServerBaseSettings = Depends(app_settings),
) -> ModerationCreateResponse:
    response = await ModerationsHandler().handle_moderations_request(
        request=request,
        moderation_request=openai_client_moderation_request[1],
        openai_client=openai_client_moderation_request[0],
        settings=settings,
    )

    # Fix type error: data missing from openai api response
    # The response types are not matching up with openai python library types
    data = (
        response.model_dump(by_alias=True)
        if isinstance(response, BaseModel)
        else response
    )
    for result in data["results"]:
        # Categories
        for k, v in Categories.model_fields.items():
            _key = v.alias or k
            if _key in result["categories"]:
                if result["categories"][_key] is None:
                    result["categories"][_key] = False
            else:
                result["categories"][_key] = False
        # CategoryScores
        for k, v in CategoryScores.model_fields.items():
            _key = v.alias or k
            if _key in result["category_scores"]:
                if result["category_scores"][_key] is None:
                    result["category_scores"][_key] = 0.0
            else:
                result["category_scores"][_key] = 0.0
        # CategoryAppliedInputTypes
        if (
            "category_applied_input_types" not in result
            or result["category_applied_input_types"] is None
        ):
            result["category_applied_input_types"] = {
                v.alias or k: []
                for k, v in CategoryAppliedInputTypes.model_fields.items()
            }

    return ModerationCreateResponse.model_validate(data)
