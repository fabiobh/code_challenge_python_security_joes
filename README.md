# MCP Weather Agent

A Python MCP HTTP server with LangGraph ReAct agent that answers: *"What is the weather forecast of the data center?"*

## Quick Start

### 1. Install Dependencies

```bash
cd code_challenge_python_security_joes
pip install -e .
```

### 2. Set Environment Variables

```bash
# Windows
set GOOGLE_API_KEY=your_api_key_here

# Linux/Mac
export GOOGLE_API_KEY=your_api_key_here
```

### 3. Start the MCP Server

```bash
python -m src.mcp_server.main
```

The server starts at http://localhost:8000

### 4. Run the Agent (in a new terminal)

```bash
python -m src.agent.main
```

### 5. Ask a Question

```
You: What is the weather forecast of the data center?
```

## Project Structure

```
├── src/
│   ├── mcp_server/     # FastAPI MCP server
│   │   ├── main.py     # Server entry point
│   │   ├── tools.py    # Tool implementations
│   │   └── models.py   # Pydantic models
│   ├── mcp_client/     # MCP client library
│   │   └── client.py   # Async client
│   └── agent/          # LangGraph ReAct agent
│       ├── main.py     # CLI entry point
│       ├── graph.py    # LangGraph nodes
│       └── state.py    # State definitions
├── docs/
│   └── ARCHITECTURE.md # Architecture docs
└── pyproject.toml      # Dependencies
```

## Tools

| Tool | Purpose | API |
|------|---------|-----|
| `ipify` | Get server's public IP | api.ipify.org |
| `ip_to_geo` | IP to coordinates | ip-api.com |
| `weather_forecast` | Weather for location | open-meteo.com |

## Testing the Server

```bash
# List tools
curl http://localhost:8000/tools

# Get public IP
curl -X POST http://localhost:8000/tools/ipify/call -H "Content-Type: application/json" -d "{}"

# Get geolocation (replace with actual IP)
curl -X POST http://localhost:8000/tools/ip_to_geo/call -H "Content-Type: application/json" -d "{\"params\": {\"ip\": \"8.8.8.8\"}}"

# Get weather
curl -X POST http://localhost:8000/tools/weather_forecast/call -H "Content-Type: application/json" -d "{\"params\": {\"latitude\": 37.7749, \"longitude\": -122.4194}}"
```

## Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for:
- System architecture
- LangGraph state & nodes
- Tool-calling strategy
- Assumptions & limitations

# use local server
set MCP_SERVER_HOST=localhost
set MCP_SERVER_PORT=8000
python -m src.agent.main

# using render.com server - located on United States, Oregon
set MCP_SERVER_HOST=https://mcp-weather-server-security-joes.onrender.com/
set MCP_SERVER_PORT=443
python -m src.agent.main