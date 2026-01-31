"""
MCP HTTP Server - FastAPI application exposing IPify, IP-API, and Open-Meteo tools.

This is the main entry point for the MCP server. It's built with FastAPI because:
1. Async support out of the box (our tools make HTTP calls, async helps performance)
2. Automatic OpenAPI docs (check /docs when running!)
3. Easy request validation with Pydantic integration

The server exposes a simple REST API:
- GET /         - Health check
- GET /tools    - List available tools (MCP discovery)
- POST /tools/{name}/call - Execute a specific tool

Key features implemented here:
- Structured JSON logging (for production debugging)
- Request ID middleware (for tracing requests across logs)
- CORS support (for browser-based clients, if needed)
"""

import logging
import sys
import json
import uuid
import contextvars
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .tools import get_tool_definitions, call_tool, TOOL_HANDLERS
from .models import ToolCallRequest, ToolCallResponse


# =============================================================================
# Request ID Context Variable
# =============================================================================
# We use a context variable to store the request ID for each request.
# This way, any log message during that request can include the ID,
# making it super easy to trace a single request through the logs.

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


# =============================================================================
# Structured Logging Setup
# =============================================================================
# Instead of plain text logs, we output JSON. This makes logs:
# - Easy to parse with log aggregation tools (Datadog, Splunk, etc.)
# - Consistent in format
# - Searchable by any field

class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON lines.
    
    Each log line includes: timestamp, level, logger name, message,
    and any extra fields we add (like request_id or tool_name).
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Include the request ID if we have one (set by middleware)
        req_id = request_id_var.get()
        if req_id:
            log_data["request_id"] = req_id
        
        # Include exception details if this is an error with traceback
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add any custom fields passed via the 'extra' parameter
        # e.g., logger.info("...", extra={"tool_name": "ipify"})
        if hasattr(record, "tool_name"):
            log_data["tool_name"] = record.tool_name
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
            
        return json.dumps(log_data)


def setup_logging():
    """
    Configure logging for the entire application.
    
    We replace the default handlers with our JSON formatter.
    This includes uvicorn's loggers so everything is consistent.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler]
    
    # Also configure uvicorn's loggers to use our format
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = [handler]


logger = logging.getLogger(__name__)


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler - runs on startup and shutdown.
    
    We set up logging here so it's ready before any requests come in.
    The 'yield' separates startup code from shutdown code.
    """
    setup_logging()
    logger.info("MCP Server starting up")
    yield
    logger.info("MCP Server shutting down")


# Create the FastAPI app with some metadata for the auto-generated docs
app = FastAPI(
    title="MCP Weather Server",
    description="MCP HTTP Server exposing tools for IP lookup, geolocation, and weather forecast",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - allows requests from any origin
# In production, you'd want to restrict this to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """
    Middleware that assigns a unique ID to each request.
    
    This ID is:
    1. Stored in a context variable (so logs can include it)
    2. Returned in the X-Request-ID response header (so clients can reference it)
    
    Super helpful for debugging - if a user reports an issue,
    they can give you the request ID and you can find exactly
    what happened in the logs.
    """
    # Generate a short unique ID (8 chars is enough for local debugging)
    req_id = str(uuid.uuid4())[:8]
    
    # Store in context variable so logger can access it
    request_id_var.set(req_id)
    
    # Process the request
    response = await call_next(request)
    
    # Add the ID to response headers
    response.headers["X-Request-ID"] = req_id
    
    return response


# =============================================================================
# MCP Endpoints
# =============================================================================
# These are the actual API endpoints that clients call.

@app.get("/")
async def root():
    """
    Health check endpoint.
    
    Returns a simple JSON showing the service is up.
    Load balancers and monitoring tools can poll this.
    """
    return {
        "status": "ok",
        "service": "mcp-weather-server",
        "version": "1.0.0",
    }


@app.get("/tools")
async def list_tools():
    """
    List all available MCP tools (MCP discovery endpoint).
    
    This is the first thing a client calls to find out what tools
    we offer. The response includes the tool name, description,
    and JSON schema for the expected parameters.
    
    The LLM agent uses this to understand what it can do.
    """
    logger.info("Listing available tools")
    tools = get_tool_definitions()
    return {"tools": tools}


@app.post("/tools/{tool_name}/call", response_model=ToolCallResponse)
async def call_tool_endpoint(tool_name: str, request: ToolCallRequest):
    """
    Execute a specific MCP tool.
    
    This is where the magic happens! The client sends:
    - tool_name in the URL path
    - parameters in the request body
    
    We execute the tool and return either:
    - {"success": true, "result": {...}} on success
    - {"success": false, "error": "..."} on failure
    
    I chose to always return 200 OK and use the success field
    instead of HTTP error codes because it simplifies client code.
    """
    import time
    start_time = time.time()
    
    # Check if the tool exists before trying to call it
    if tool_name not in TOOL_HANDLERS:
        logger.warning(f"Unknown tool requested: {tool_name}")
        raise HTTPException(
            status_code=404,
            detail=f"Tool not found: {tool_name}. Available tools: {list(TOOL_HANDLERS.keys())}"
        )
    
    logger.info(f"Calling tool: {tool_name}", extra={"tool_name": tool_name})
    
    try:
        # Execute the tool with the provided parameters
        result = await call_tool(tool_name, request.params)
        
        # Log how long it took (useful for performance monitoring)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Tool {tool_name} completed successfully",
            extra={"tool_name": tool_name, "duration_ms": round(duration_ms, 2)}
        )
        
        return ToolCallResponse(success=True, result=result)
        
    except ValidationError as e:
        # Pydantic validation failed (bad parameters)
        logger.error(f"Validation error for tool {tool_name}: {e}")
        return ToolCallResponse(
            success=False,
            error=f"Invalid parameters: {str(e)}"
        )
        
    except ValueError as e:
        # Business logic error (e.g., invalid IP from IP-API)
        logger.error(f"Value error for tool {tool_name}: {e}")
        return ToolCallResponse(
            success=False,
            error=str(e)
        )
        
    except Exception as e:
        # Unexpected error - log the full traceback
        logger.exception(f"Unexpected error calling tool {tool_name}")
        return ToolCallResponse(
            success=False,
            error=f"Internal error: {str(e)}"
        )


# =============================================================================
# Main Entry Point
# =============================================================================
# This runs when you execute: python -m src.mcp_server.main

def main():
    """
    Run the MCP server using uvicorn.
    
    Host and port can be configured via environment variables:
    - MCP_SERVER_HOST (default: localhost)
    - MCP_SERVER_PORT (default: 8000)
    """
    import uvicorn
    import os
    
    host = os.getenv("MCP_SERVER_HOST", "localhost")
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    
    # Print a nice startup message
    print(f"\n🚀 Starting MCP Weather Server at http://{host}:{port}")
    print(f"📋 Available tools: ipify, ip_to_geo, weather_forecast")
    print(f"📖 API docs: http://{host}:{port}/docs\n")
    
    uvicorn.run(
        "src.mcp_server.main:app",
        host=host,
        port=port,
        reload=False,
        log_config=None,  # Disable uvicorn's default logging, we use our own
    )


if __name__ == "__main__":
    main()
