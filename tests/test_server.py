"""
Integration tests for MCP Server API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.mcp_server.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthCheck:
    """Tests for health check endpoint."""
    
    def test_health_check_returns_ok(self, client):
        """Test that health check returns ok status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
        assert "version" in data


class TestListTools:
    """Tests for tool listing endpoint."""
    
    def test_list_tools_returns_three_tools(self, client):
        """Test that /tools returns exactly 3 tools."""
        response = client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) == 3
    
    def test_list_tools_has_required_fields(self, client):
        """Test that each tool has name, description, input_schema."""
        response = client.get("/tools")
        data = response.json()
        
        for tool in data["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
    
    def test_list_tools_names(self, client):
        """Test that the correct tool names are present."""
        response = client.get("/tools")
        data = response.json()
        
        tool_names = [t["name"] for t in data["tools"]]
        assert "ipify" in tool_names
        assert "ip_to_geo" in tool_names
        assert "weather_forecast" in tool_names


class TestToolCallValidation:
    """Tests for tool call input validation."""
    
    def test_unknown_tool_returns_404(self, client):
        """Test that unknown tool name returns 404."""
        response = client.post("/tools/unknown_tool/call", json={"params": {}})
        assert response.status_code == 404
    
    def test_ip_to_geo_missing_ip_returns_error(self, client):
        """Test that missing IP parameter returns error."""
        response = client.post("/tools/ip_to_geo/call", json={"params": {}})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    def test_ip_to_geo_invalid_ip_returns_error(self, client):
        """Test that invalid IP format returns error."""
        response = client.post("/tools/ip_to_geo/call", json={"params": {"ip": "invalid"}})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
    
    def test_weather_forecast_missing_params_returns_error(self, client):
        """Test that missing coordinates return error."""
        response = client.post("/tools/weather_forecast/call", json={"params": {}})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
    
    def test_weather_forecast_invalid_latitude_returns_error(self, client):
        """Test that invalid latitude returns error."""
        response = client.post(
            "/tools/weather_forecast/call",
            json={"params": {"latitude": 100, "longitude": 0}}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestRequestIdHeader:
    """Tests for request ID tracing."""
    
    def test_response_has_request_id_header(self, client):
        """Test that responses include X-Request-ID header."""
        response = client.get("/")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8
    
    def test_request_id_is_unique(self, client):
        """Test that each request gets a unique ID."""
        ids = set()
        for _ in range(5):
            response = client.get("/")
            ids.add(response.headers["X-Request-ID"])
        
        assert len(ids) == 5  # All unique
