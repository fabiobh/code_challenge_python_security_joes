"""
MCP Client - Python client for consuming the MCP HTTP server.

This is the agent's gateway to the MCP server. Instead of having the agent
make raw HTTP calls, we wrap everything in a nice async client class.

Benefits of having a dedicated client:
1. Encapsulates all the HTTP logic in one place
2. Handles errors consistently
3. Provides tool caching (don't fetch the tool list every time)
4. Makes the agent code cleaner and easier to test

The client is async because the agent runs in an async context,
and we don't want to block while waiting for the server to respond.
"""

import httpx
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================
# Having specific exception types makes error handling much cleaner.
# The agent can catch MCPToolError vs MCPClientError and handle them differently.

class MCPClientError(Exception):
    """
    Base exception for MCP client errors.
    
    This covers connection issues, HTTP errors, etc.
    Basically anything that prevents us from talking to the server.
    """
    pass


class MCPToolError(MCPClientError):
    """
    Exception raised when a tool call fails.
    
    This is different from MCPClientError - the connection worked,
    but the tool itself returned an error (e.g., invalid IP address).
    """
    pass


class MCPClient:
    """
    Async client for interacting with the MCP HTTP server.
    
    Usage:
        client = MCPClient("http://localhost:8000")
        await client.list_tools()  # Discover available tools
        result = await client.call_tool("ipify", {})  # Call a tool
    
    The client caches the tool list after the first fetch,
    so subsequent calls are faster.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """
        Initialize the MCP client.
        
        Args:
            base_url: Where the MCP server is running
            timeout: How long to wait for responses (in seconds)
        """
        self.base_url = base_url.rstrip("/")  # Remove trailing slash if present
        self.timeout = timeout
        self._tools_cache: Optional[list[dict]] = None  # Cached tool definitions
        
        logger.info(f"MCP Client initialized with base URL: {self.base_url}")
    
    async def health_check(self) -> bool:
        """
        Check if the MCP server is running and healthy.
        
        This is a quick ping to verify connectivity before we start
        making tool calls. Returns True if server is up, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/")
                return response.status_code == 200 and response.json().get("status") == "ok"
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def list_tools(self, use_cache: bool = True) -> list[dict]:
        """
        Fetch the list of available tools from the MCP server.
        
        This is what tells the LLM what tools it can use. The response
        includes tool names, descriptions, and parameter schemas.
        
        Args:
            use_cache: If True, return cached list if we have it.
                      Set to False to force a fresh fetch.
        
        Returns:
            List of tool definitions (name, description, input_schema)
        """
        # Return cached tools if we have them (saves an HTTP call)
        if use_cache and self._tools_cache is not None:
            return self._tools_cache
        
        logger.info("Fetching tool list from MCP server")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/tools")
                response.raise_for_status()  # Raise on 4xx/5xx
                
                data = response.json()
                self._tools_cache = data.get("tools", [])
                
                logger.info(f"Fetched {len(self._tools_cache)} tools")
                return self._tools_cache
                
        except httpx.HTTPStatusError as e:
            raise MCPClientError(f"Failed to fetch tools: HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise MCPClientError(f"Failed to connect to MCP server: {e}") from e
    
    async def call_tool(self, name: str, params: Optional[dict] = None) -> Any:
        """
        Call a specific tool on the MCP server.
        
        This is where we actually execute a tool. The agent decides
        which tool to call and with what params, then we make it happen.
        
        Args:
            name: Tool name (e.g., "ipify", "ip_to_geo", "weather_forecast")
            params: Tool parameters (varies by tool)
        
        Returns:
            The tool's result (structure depends on the tool)
        
        Raises:
            MCPToolError: If the tool returns an error
            MCPClientError: If we can't reach the server
        """
        params = params or {}  # Default to empty dict if None
        
        logger.info(f"Calling tool: {name} with params: {params}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/tools/{name}/call",
                    json={"params": params},
                )
                
                # Handle 404 specially - tool doesn't exist
                if response.status_code == 404:
                    raise MCPToolError(f"Tool not found: {name}")
                
                response.raise_for_status()
                
                data = response.json()
                
                # Check if the tool execution succeeded
                # (server returns 200 but with success=false on tool errors)
                if not data.get("success", False):
                    error_msg = data.get("error", "Unknown error")
                    raise MCPToolError(f"Tool '{name}' failed: {error_msg}")
                
                result = data.get("result")
                logger.info(f"Tool {name} returned: {result}")
                
                return result
                
        except httpx.HTTPStatusError as e:
            raise MCPClientError(f"Tool call failed: HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise MCPClientError(f"Failed to connect to MCP server: {e}") from e
    
    def get_tools_for_llm(self) -> list[dict]:
        """
        Get tool definitions formatted for LLM function calling.
        
        This transforms our tool definitions into the format expected
        by LLM function calling APIs (like OpenAI's or Google's).
        
        Returns:
            List of tools in function-calling format
        """
        if self._tools_cache is None:
            return []
        
        # Transform to the standard function calling format
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                }
            }
            for tool in self._tools_cache
        ]
    
    def format_tool_for_prompt(self) -> str:
        """
        Format tool descriptions as text for inclusion in prompts.
        
        This is an alternative to function calling - we can just include
        the tool descriptions in the system prompt as plain text.
        The LLM then outputs JSON to call tools.
        
        Returns:
            Human-readable description of available tools
        """
        if self._tools_cache is None:
            return "No tools available."
        
        lines = ["Available tools:"]
        for tool in self._tools_cache:
            lines.append(f"\n- **{tool['name']}**: {tool['description']}")
            
            # Add parameter documentation
            schema = tool.get("input_schema", {})
            props = schema.get("properties", {})
            required = schema.get("required", [])
            
            if props:
                lines.append("  Parameters:")
                for param_name, param_info in props.items():
                    req_marker = " (required)" if param_name in required else ""
                    desc = param_info.get("description", "")
                    lines.append(f"    - {param_name}{req_marker}: {desc}")
        
        return "\n".join(lines)
