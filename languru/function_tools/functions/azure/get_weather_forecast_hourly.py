import json
import os
from datetime import datetime
from textwrap import dedent
from typing import Any, Callable, ClassVar, Dict, Text

import requests
from json_repair import repair_json
from pydantic import Field, field_validator
from pydantic_core import ValidationError

from languru.config import logger
from languru.function_tools.function_base_model import FunctionToolRequestBaseModel

AZURE_MAPS_KEY = os.environ["AZURE_MAPS_KEY"]
AZURE_WEATHER_FORECAST_HOURLY_URL = (
    "https://atlas.microsoft.com/weather/forecast/hourly/json"
)
AZURE_WEATHER_FORECAST_HOURLY_API_VERSION = "1.1"


def get_weather_forecast_hourly(request: "GetWeatherForecastHourly"):
    url = AZURE_WEATHER_FORECAST_HOURLY_URL
    params = request.model_dump(exclude_none=True)
    params["api-version"] = AZURE_WEATHER_FORECAST_HOURLY_API_VERSION
    params["subscription-key"] = AZURE_MAPS_KEY
    response = requests.get(url, params=params)
    return response.json()


def format_weather_article(weather_data: Dict[Text, Any]) -> Text:
    article_parts = []

    # Process each forecast
    for forecast in weather_data.get("forecasts", []):
        try:
            # Parse date
            date_str = forecast.get("date")
            if date_str:
                date = datetime.fromisoformat(date_str)
                date_formatted = date.strftime("%A, %B %d, %Y %I:%M %p")
                article_parts.append(f"\n=== {date_formatted} ===")

            # Weather conditions
            icon_phrase = forecast.get("iconPhrase", "Unknown")
            article_parts.append(f"Conditions: {icon_phrase}")

            # Temperature and Real Feel
            temp = forecast.get("temperature", {}).get("value")
            real_feel = forecast.get("realFeelTemperature", {}).get("value")
            if temp is not None:
                article_parts.append(f"Temperature: {temp}°C")
            if real_feel is not None:
                article_parts.append(f"Feels like: {real_feel}°C")

            # Wind details
            wind = forecast.get("wind", {})
            direction = wind.get("direction", {}).get("localizedDescription", "Unknown")
            speed = wind.get("speed", {}).get("value")
            if speed is not None:
                article_parts.append(f"Wind: {direction} at {speed} km/h")

            # Precipitation details
            has_precipitation = forecast.get("hasPrecipitation", False)
            if has_precipitation:
                precip_type = forecast.get("precipitationType", "Unknown")
                precip_intensity = forecast.get("precipitationIntensity", "Unknown")
                article_parts.append(
                    f"Precipitation: {precip_type} ({precip_intensity})"
                )

            # Additional details
            humidity = forecast.get("relativeHumidity")
            if humidity is not None:
                article_parts.append(f"Humidity: {humidity}%")

            visibility = forecast.get("visibility", {}).get("value")
            if visibility is not None:
                article_parts.append(f"Visibility: {visibility} km")

            cloud_cover = forecast.get("cloudCover")
            if cloud_cover is not None:
                article_parts.append(f"Cloud Cover: {cloud_cover}%")

        except Exception as e:
            logger.exception(e)
            continue  # Skip problematic forecasts

    return "\n".join(article_parts).strip()


class GetWeatherForecastHourly(FunctionToolRequestBaseModel):
    FUNCTION_NAME: ClassVar[Text] = "get_weather_forecast_hourly"
    FUNCTION_DESCRIPTION: ClassVar[Text] = dedent(
        """
        This function retrieves an hourly weather forecast from the Azure Maps Weather API.
        It accepts parameters such as location query, duration, and language to customize the forecast data.
        The function returns detailed weather information including temperature, humidity, wind, precipitation, and UV index for the specified duration.
        """  # noqa: E501
    ).strip()
    FUNCTION: ClassVar[Callable[["GetWeatherForecastHourly"], Dict]] = (
        get_weather_forecast_hourly
    )

    query: Text = Field(
        ...,
        description=dedent(
            """
            The location query specified as a comma-separated string composed of latitude followed by longitude, e.g., "47.641268,-122.125679".
            This parameter is required to identify the geographical location for which the weather forecast is requested.
            """  # noqa: E501
        ).strip(),
    )
    duration: int = Field(
        default=1,
        description=dedent(
            """
            The time frame for the returned weather forecast. Available values are:
            1 - Return forecast data for the next hour (default).
            12 - Return hourly forecast for the next 12 hours.
            24 - Return hourly forecast for the next 24 hours.
            72 - Return hourly forecast for the next 72 hours (3 days).
            """  # noqa: E501
        ).strip(),
    )
    language: Text = Field(
        default="en",
        description=dedent(
            """
            The language in which the search results should be returned. Should be one of the supported IETF language tags, case insensitive.
            When data in the specified language is not available for a specific field, the default language is used.
            """  # noqa: E501
        ).strip(),
    )

    @field_validator("query")
    def validate_query(cls, v: Text) -> Text:
        v = v.strip()
        if not v:
            raise ValidationError("Parameter `query` cannot be empty")
        return v

    @classmethod
    def parse_response_as_tool_content(cls, response: Dict[Text, Any]) -> Text:
        if not response and not isinstance(response, Dict):
            return cls.FUNCTION_ERROR_CONTENT
        try:
            return format_weather_article(response)
        except Exception as e:
            logger.exception(e)
            return cls.FUNCTION_ERROR_CONTENT

    @classmethod
    def from_args_str(cls, args_str: Text):
        func_kwargs = (
            json.loads(repair_json(args_str)) if args_str else {}  # type: ignore
        )

        return cls.model_validate(func_kwargs)
