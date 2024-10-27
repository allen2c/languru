import os
from textwrap import dedent
from typing import Callable, ClassVar, Literal, Text

import requests
from pydantic import Field

from languru.function_tools.function_base_model import FunctionToolRequestBaseModel

AZURE_MAPS_KEY = os.environ["AZURE_MAPS_KEY"]
AZURE_WEATHER_FORECAST_DAILY_URL = (
    "https://atlas.microsoft.com/weather/forecast/daily/json"
)
AZURE_WEATHER_FORECAST_DAILY_API_VERSION = "1.1"


def get_weather_forecast_daily(request: "GetWeatherForecastDaily"):
    url = AZURE_WEATHER_FORECAST_DAILY_URL
    params = request.model_dump(exclude_none=True)
    params["api-version"] = AZURE_WEATHER_FORECAST_DAILY_API_VERSION
    params["subscription-key"] = AZURE_MAPS_KEY
    response = requests.get(url, params=params)
    return response.json()


class GetWeatherForecastDaily(FunctionToolRequestBaseModel):
    FUNCTION_NAME: ClassVar[Text] = "get_weather_forecast_daily"
    FUNCTION_DESCRIPTION: ClassVar[Text] = dedent(
        """
        The Get Daily Forecast API is an HTTP GET request that returns detailed weather forecast such as temperature and wind by day for the next 1, 5, or 10 days for a given coordinate location. The response includes details such as temperature, wind, precipitation, air quality, and UV index
        """  # noqa: E501
    ).strip()
    FUNCTION: ClassVar[Callable] = get_weather_forecast_daily

    query: Text = Field(
        ...,
        description=dedent(
            """
            The applicable query specified as a comma separated string composed by latitude followed by longitude e.g. "47.641268,-122.125679".
            Weather information is generally available for locations on land, bodies of water surrounded by land, and areas of the ocean that are within approximately 50 nautical miles of a coastline.
            """  # noqa: E501
        ).strip(),
    )
    duration: Literal[1, 5, 10] = Field(
        default=1,
        description=dedent(
            """
            Specifies for how many days the daily forecast responses are returned. Available values are:
            1 - Return forecast data for the next day. Returned by default.
            5 - Return forecast data for the next 5 days.
            10 - Return forecast data for the next 10 days.
            """  # noqa: E501
        ).strip(),
    )
    language: Text = Field(
        default="en",
        description=dedent(
            """
            The applicable query specified as a comma separated string composed by latitude followed by longitude e.g. "47.641268,-122.125679".
            Weather information is generally available for locations on land, bodies of water surrounded by land, and areas of the ocean that are within approximately 50 nautical miles of a coastline.
            """  # noqa: E501
        ).strip(),
    )
