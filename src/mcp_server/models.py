"""
Pydantic models for MCP Server input validation and response structures.

This file defines all the data models used across the MCP server. I'm using Pydantic
here because it gives us automatic validation, serialization, and nice error messages
out of the box. Much better than manually checking everything!

The models are organized into sections:
1. MCP Protocol models - generic structures for the MCP server/client communication
2. Tool-specific models - request/response models for each external API we call
"""

from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator
import ipaddress


# =============================================================================
# MCP Protocol Models
# =============================================================================
# These are the core models that define how clients talk to our MCP server.
# They're tool-agnostic - any tool can use these structures.

class ToolDefinition(BaseModel):
    """
    Describes what a tool does and what parameters it expects.
    
    This is what we return when a client asks "what tools do you have?"
    The LLM uses this info to decide which tool to call and with what params.
    """
    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Human-readable description")
    # JSON Schema format - this tells the LLM exactly what params are expected
    input_schema: dict = Field(..., description="JSON Schema for tool inputs")


class ToolCallRequest(BaseModel):
    """
    The request body when a client wants to execute a tool.
    
    Pretty simple - just a dict of parameters. The params vary by tool,
    so we keep this generic and let each tool validate its own params.
    """
    params: dict = Field(default_factory=dict, description="Tool parameters")


class ToolCallResponse(BaseModel):
    """
    Standard response format for all tool calls.
    
    I went with a success/error pattern here instead of throwing HTTP errors
    because it makes the client code simpler - they always get a 200 OK and
    can check the success field to know if the tool worked.
    """
    success: bool = Field(..., description="Whether the call succeeded")
    result: Optional[Any] = Field(None, description="Tool result on success")
    error: Optional[str] = Field(None, description="Error message on failure")


# =============================================================================
# IPify Models
# =============================================================================
# IPify is dead simple - no input, just returns our public IP.

class IPifyResponse(BaseModel):
    """Response from the ipify API - just an IP address string."""
    ip: str = Field(..., description="Public IP address")


# =============================================================================
# IP-API (Geolocation) Models
# =============================================================================
# These handle the IP-to-location lookup. The validation here is important
# because we don't want to hit the API with garbage IPs.

class IPToGeoRequest(BaseModel):
    """
    Request to convert an IP address to geographic coordinates.
    
    The validator below ensures we only accept valid IPv4 or IPv6 addresses.
    This saves us from making pointless API calls with invalid data.
    """
    ip: str = Field(..., description="IP address to geolocate")
    
    @field_validator("ip")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        """
        Validate IP address format using Python's ipaddress module.
        
        I switched from regex to ipaddress because regex was getting messy
        for IPv6, and the stdlib module handles all the edge cases properly.
        """
        if not v or not v.strip():
            raise ValueError("IP address cannot be empty")
        
        try:
            # This handles both IPv4 and IPv6 with full validation
            # It checks octets are in range, IPv6 syntax is correct, etc.
            ipaddress.ip_address(v)
            return v
        except ValueError:
            # Re-raise with a cleaner error message
            raise ValueError(f"Invalid IP address: {v}")


class GeoLocation(BaseModel):
    """
    Response from the IP-API geolocation service.
    
    The API returns a lot of fields, but we only care about a few of them.
    I made most fields optional because the API doesn't always return everything
    (especially if the lookup fails or the IP is internal).
    """
    status: str = Field(..., description="API response status")
    country: Optional[str] = Field(None, description="Country name")
    city: Optional[str] = Field(None, description="City name")
    lat: Optional[float] = Field(None, description="Latitude")
    lon: Optional[float] = Field(None, description="Longitude")
    message: Optional[str] = Field(None, description="Error message if failed")


# =============================================================================
# Open-Meteo (Weather) Models
# =============================================================================
# Weather API models. The coordinate validation is strict because
# the API will error on invalid coords anyway.

class WeatherForecastRequest(BaseModel):
    """
    Request for weather forecast at specific coordinates.
    
    The ge/le validators ensure lat is in [-90, 90] and lon is in [-180, 180].
    These are hard physical limits - there's no point accepting invalid coords.
    """
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")


class CurrentWeather(BaseModel):
    """
    Current weather conditions from Open-Meteo.
    
    All fields optional because the API response structure can vary.
    We just grab what's available.
    """
    temperature_2m: Optional[float] = Field(None, description="Temperature at 2m height (°C)")
    weather_code: Optional[int] = Field(None, description="WMO weather code")
    wind_speed_10m: Optional[float] = Field(None, description="Wind speed at 10m (km/h)")
    relative_humidity_2m: Optional[int] = Field(None, description="Relative humidity (%)")


class WeatherForecast(BaseModel):
    """Full weather forecast response - we don't use this directly but it's good to have."""
    latitude: float
    longitude: float
    timezone: Optional[str] = None
    current: Optional[CurrentWeather] = None
    error: Optional[bool] = None
    reason: Optional[str] = None


# =============================================================================
# WMO Weather Code Descriptions
# =============================================================================
# The Open-Meteo API returns numeric weather codes (WMO standard).
# This lookup table converts them to human-readable descriptions.
# I got these from the WMO documentation.

WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_weather_description(code: Optional[int]) -> str:
    """
    Convert a WMO weather code to a human-readable description.
    
    Returns "Unknown" if the code is None or not in our lookup table.
    This way the agent can always show something meaningful to the user.
    """
    if code is None:
        return "Unknown"
    return WMO_WEATHER_CODES.get(code, f"Unknown code: {code}")
