import os
from textwrap import dedent
from typing import Any, Callable, ClassVar, Dict, List, Optional, Text

import googlemaps
from pydantic import BaseModel, Field
from pydantic_core import ValidationError

from languru.config import logger
from languru.function_tools.function_base_model import FunctionToolRequestBaseModel

# Initialize the client with your API key
gmaps = googlemaps.Client(key=os.environ["GEOCODING_API_KEY"])


# Get coordinates for a place
def get_maps_geocode(request: "GetMapsGeocode") -> Optional[Dict[Text, Any]]:
    try:
        # Geocoding request
        results = gmaps.geocode(  # type: ignore
            address=request.address, region=request.region, language=request.language
        )
        return {"results": results, "status": "ok"}
    except Exception as e:
        logger.exception(e)
        logger.error(f"Error getting coordinates for {request.address}")
        return None


def format_maps_geocode_article(geocode_data: Optional[Dict[Text, Any]]) -> Text:
    if geocode_data is None:
        return "No maps geocode results found."

    try:
        geocode_res = GeocodeResponse.model_validate(geocode_data)
        results = geocode_res.model_dump()["results"] or []
    except ValidationError as e:
        logger.exception(e)
        logger.error(f"Error validating maps geocode data: {geocode_data}")
        results = geocode_data.get("results", []) or []

    if not results:
        return "No maps geocode results found."

    article_parts = []
    for result in results:
        formatted_address = result.get("formatted_address", "N/A")
        types = ", ".join(result.get("types", []))

        # Extract important address components
        address_components = result.get("address_components", [])
        important_components = [
            component["long_name"]
            for component in address_components
            if "country" in component["types"] or "locality" in component["types"]
        ]
        important_components_str = ", ".join(important_components)

        # Extract geometry information
        geometry = result.get("geometry", {})
        location = geometry.get("location", {})
        lat = location.get("lat", "N/A")
        lng = location.get("lng", "N/A")

        article_parts.append(
            f"The most similar or closest address '{formatted_address}' is located in {important_components_str}. "  # noqa: E501
            f"It is positioned at latitude {lat} and longitude {lng}. "
            f"This location is categorized as: {types}."
        )

    return "\n".join([f"<place>{p}</place>" for p in article_parts if p])


class GetMapsGeocode(FunctionToolRequestBaseModel):
    FUNCTION_NAME: ClassVar[Text] = "get_maps_geocode"
    FUNCTION_DESCRIPTION: ClassVar[Text] = dedent(
        """
        This function retrieves an hourly weather forecast from the Azure Maps Weather API.
        It accepts parameters such as location query, duration, and language to customize the forecast data.
        The function returns detailed weather information including temperature, humidity, wind, precipitation, and UV index for the specified duration.
        """  # noqa: E501
    ).strip()
    FUNCTION: ClassVar[Callable[["GetMapsGeocode"], Optional[Dict]]] = get_maps_geocode

    address: Optional[Text] = Field(..., description="The address to geocode.")
    region: Optional[Text] = Field(
        default=None,
        description="The region code, specified as a ccTLD ('top-level domain') two-character value.",  # noqa: E501
    )
    language: Optional[Text] = Field(
        default=None,
        description="The language in which to return results.",
    )

    @classmethod
    def parse_response_as_tool_content(
        cls, response: Optional[Dict[Text, Any]]
    ) -> Text:
        if not response and not isinstance(response, Dict):
            return cls.FUNCTION_ERROR_CONTENT
        try:
            return format_maps_geocode_article(response)
        except Exception as e:
            logger.exception(e)
            return cls.FUNCTION_ERROR_CONTENT


class AddressComponent(BaseModel):
    long_name: Text
    short_name: Text
    types: List[Text] = Field(default_factory=list)


class Location(BaseModel):
    lat: float
    lng: float


class Bounds(BaseModel):
    northeast: Location
    southwest: Location


class Viewport(BaseModel):
    northeast: Location
    southwest: Location


class Geometry(BaseModel):
    bounds: Optional[Bounds] = Field(default=None)
    location: Location
    location_type: Text
    viewport: Viewport = Field(default=None)


class PlusCode(BaseModel):
    compound_code: Optional[Text] = Field(default=None)
    global_code: Text


class GeocodeResult(BaseModel):
    address_components: List[AddressComponent]
    formatted_address: Text
    geometry: Geometry
    partial_match: Optional[bool] = Field(default=None)
    place_id: Text
    plus_code: Optional[PlusCode] = Field(default=None)
    types: List[Text]


class GeocodeResponse(BaseModel):
    results: List[GeocodeResult]
    status: Text = Field(default="ok", description="The status of the request.")
