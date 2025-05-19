import asyncio
import datetime
import typing
import zoneinfo

import agents
import pydantic
from json_repair import json_repair


class GetTimeNow:
    name = "GetTimeNow"
    description = "Get the current time. This tool may take very very long time to execute, so you must inform the user to wait patiently, then emit tool call request."  # noqa: E501

    @staticmethod
    async def get_time_now(ctx: agents.RunContextWrapper[typing.Any], args: str) -> str:
        await asyncio.sleep(0)
        args_model = GetTimeNow.GetTimeNowArgs.model_validate(json_repair.loads(args))

        dt_now = datetime.datetime.now(zoneinfo.ZoneInfo(args_model.timezone))
        dt_now = dt_now.replace(microsecond=0)
        return dt_now.isoformat()

    class GetTimeNowArgs(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")

        timezone: str = pydantic.Field(
            default="Asia/Taipei",
            description="Optional. The timezone for the current time. Default is Asia/Taipei.",  # noqa: E501
        )
