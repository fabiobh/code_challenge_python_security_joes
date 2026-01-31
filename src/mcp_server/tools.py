"""
MCP Tool implementations for IPify, IP-API, and Open-Meteo.

This is where the actual work happens! Each tool is basically a wrapper around
an external API. The tools follow a simple pattern:
1. Validate input parameters (using Pydantic models)
2. Make the HTTP request to the external API
3. Parse and return the response

I added retry logic with exponential backoff because external APIs can be flaky.
The tenacity library makes this really clean - just a decorator on each function.
"""

import httpx
import logging
from typing import Any

# Tenacity is a great retry library - handles backoff, exceptions, logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .models import (
    ToolDefinition,
    IPifyResponse,
    IPToGeoRequest,
    GeoLocation,
    WeatherForecastRequest,
    WeatherForecast,
    CurrentWeather,
    get_weather_description,
)

logger = logging.getLogger(__name__)

# =============================================================================
# API Endpoints
# =============================================================================
# I keep these as constants at the top so they're easy to find and change.
# Note: IP-API uses HTTP (not HTTPS) on the free tier - that's a limitation.

IPIFY_URL = "https://api.ipify.org?format=json"
IP_API_URL = "http://ip-api.com/json/{ip}"  # Free tier is HTTP only
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# =============================================================================
# Retry Configuration
# =============================================================================
# These settings control how we handle transient failures.
# 3 attempts with exponential backoff (1s, 2s, 4s... up to 10s max)
# should handle most temporary network hiccups.

RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT = 1   # Start with 1 second wait
RETRY_MAX_WAIT = 10  # Cap at 10 seconds (don't want to wait forever)


def create_retry_decorator():
    """
    Create a reusable retry decorator for external API calls.
    
    This catches network errors and timeouts, but NOT validation errors.
    We don't want to retry if the user gave us bad input - that would
    just fail again and waste time.
    """
    return retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        # Only retry on network/timeout issues, not on bad requests
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        # Log a warning before each retry so we can see what's happening
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # Raise the final exception if all retries fail
    )


# Create the decorator once and reuse it
api_retry = create_retry_decorator()


# =============================================================================
# Tool Definitions
# =============================================================================
# These definitions tell clients (and the LLM) what tools are available.
# The input_schema follows JSON Schema format - this is a standard that
# LLMs understand well, so they can generate correct tool calls.

TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="ipify",
        description="Get the public IP address of the machine running the MCP server",
        input_schema={
            "type": "object",
            "properties": {},  # No inputs needed!
            "required": [],
        },
    ),
    ToolDefinition(
        name="ip_to_geo",
        description="Get the geographical location (latitude/longitude) for a given IP address",
        input_schema={
            "type": "object",
            "properties": {
                "ip": {
                    "type": "string",
                    "description": "The IP address to geolocate (IPv4 or IPv6)",
                }
            },
            "required": ["ip"],  # IP is mandatory
        },
    ),
    ToolDefinition(
        name="weather_forecast",
        description="Get the current weather forecast for a given latitude and longitude",
        input_schema={
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude (-90 to 90)",
                    "minimum": -90,
                    "maximum": 90,
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude (-180 to 180)",
                    "minimum": -180,
                    "maximum": 180,
                },
            },
            "required": ["latitude", "longitude"],  # Both coords needed
        },
    ),
]


def get_tool_definitions() -> list[dict]:
    """
    Get all tool definitions as dictionaries for the API response.
    
    We use model_dump() to convert Pydantic models to dicts.
    This is what gets returned when someone calls GET /tools.
    """
    return [tool.model_dump() for tool in TOOL_DEFINITIONS]


# =============================================================================
# Tool Implementations
# =============================================================================
# Each tool is an async function that calls an external API.
# The @api_retry decorator handles automatic retries on failure.

@api_retry
async def call_ipify() -> dict[str, Any]:
    """
    Get the server's public IP address using the IPify API.
    
    This is the simplest tool - no parameters, just returns our IP.
    The LLM uses this as the first step to figure out where the 
    "data center" (this server) is located.
    
    Returns:
        {"ip": "123.45.67.89"} - the public IP address
    """
    logger.info("Calling IPify API to get public IP")
    
    # httpx is like requests but async - great for FastAPI
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(IPIFY_URL)
        response.raise_for_status()  # Raises on 4xx/5xx
        
        data = response.json()
        # Validate the response matches our expected structure
        result = IPifyResponse(**data)
        
        logger.info(f"IPify returned IP: {result.ip}")
        return result.model_dump()


@api_retry
async def call_ip_to_geo(params: dict) -> dict[str, Any]:
    """
    Convert an IP address to geographic coordinates using IP-API.
    
    This is step 2 in our flow - we take the IP from step 1 and
    get the lat/long coordinates. These are approximate (based on
    ISP location) but good enough for weather purposes.
    
    Args:
        params: {"ip": "123.45.67.89"}
    
    Returns:
        {"country": "Brazil", "city": "Sao Paulo", "latitude": -23.5, "longitude": -46.6}
    """
    # Validate the IP address format before making the API call
    request = IPToGeoRequest(**params)
    
    logger.info(f"Calling IP-API for IP: {request.ip}")
    
    # The IP goes in the URL path, not as a query param
    url = IP_API_URL.format(ip=request.ip)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        
        data = response.json()
        result = GeoLocation(**data)
        
        # IP-API returns status: "fail" on error instead of HTTP error codes
        # So we have to check this manually
        if result.status == "fail":
            error_msg = result.message or "Unknown error from IP-API"
            logger.error(f"IP-API failed: {error_msg}")
            raise ValueError(f"IP-API error: {error_msg}")
        
        logger.info(f"IP-API returned: {result.city}, {result.country} "
                   f"(lat={result.lat}, lon={result.lon})")
        
        # Return a cleaned-up response with just the fields we need
        return {
            "country": result.country,
            "city": result.city,
            "latitude": result.lat,
            "longitude": result.lon,
        }


@api_retry
async def call_weather_forecast(params: dict) -> dict[str, Any]:
    """
    Get weather forecast for given coordinates using Open-Meteo.
    
    This is the final step - we use the coordinates from step 2
    to get the current weather at that location. Open-Meteo is
    nice because it's free and doesn't require an API key.
    
    Args:
        params: {"latitude": -23.5, "longitude": -46.6}
    
    Returns:
        Weather data including temperature, conditions, wind, humidity
    """
    # Validate coordinates are in valid ranges
    request = WeatherForecastRequest(**params)
    
    logger.info(f"Calling Open-Meteo for lat={request.latitude}, lon={request.longitude}")
    
    # Build the query parameters for the weather API
    # We're asking for current conditions only (not the full forecast)
    query_params = {
        "latitude": request.latitude,
        "longitude": request.longitude,
        # These are the data points we want - Open-Meteo has many more available
        "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
        "timezone": "auto",  # Let the API figure out the timezone
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(OPEN_METEO_URL, params=query_params)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if the API returned an error (it uses a flag, not HTTP codes)
        if data.get("error"):
            error_msg = data.get("reason", "Unknown error from Open-Meteo")
            logger.error(f"Open-Meteo failed: {error_msg}")
            raise ValueError(f"Open-Meteo error: {error_msg}")
        
        # Parse the current weather section
        current_data = data.get("current", {})
        current = CurrentWeather(
            temperature_2m=current_data.get("temperature_2m"),
            weather_code=current_data.get("weather_code"),
            wind_speed_10m=current_data.get("wind_speed_10m"),
            relative_humidity_2m=current_data.get("relative_humidity_2m"),
        )
        
        # Convert the numeric weather code to something readable
        weather_description = get_weather_description(current.weather_code)
        
        logger.info(f"Open-Meteo returned: {current.temperature_2m}°C, {weather_description}")
        
        # Return a nice, clean response with all the weather info
        return {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "temperature_celsius": current.temperature_2m,
            "weather_code": current.weather_code,
            "weather_description": weather_description,
            "wind_speed_kmh": current.wind_speed_10m,
            "humidity_percent": current.relative_humidity_2m,
        }


# =============================================================================
# Tool Router
# =============================================================================
# This maps tool names to their handler functions.
# When a request comes in for a specific tool, we look it up here.

TOOL_HANDLERS = {
    "ipify": call_ipify,
    "ip_to_geo": call_ip_to_geo,
    "weather_forecast": call_weather_forecast,
}


async def call_tool(name: str, params: dict) -> dict[str, Any]:
    """
    Route a tool call to the appropriate handler function.
    
    This is the main entry point for executing tools. The MCP server
    endpoint calls this with the tool name and parameters.
    
    Args:
        name: Tool name (e.g., "ipify", "ip_to_geo", "weather_forecast")
        params: Tool parameters (can be empty for ipify)
    
    Returns:
        The tool's result as a dictionary
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    
    # ipify is special - it doesn't take any parameters
    if name == "ipify":
        return await handler()
    else:
        return await handler(params)
