"""
Unit tests for MCP Server models and tools.
"""

import pytest
from pydantic import ValidationError


class TestIPToGeoRequest:
    """Tests for IP address validation."""
    
    def test_valid_ipv4_simple(self):
        """Test valid simple IPv4 address."""
        from src.mcp_server.models import IPToGeoRequest
        req = IPToGeoRequest(ip="8.8.8.8")
        assert req.ip == "8.8.8.8"
    
    def test_valid_ipv4_edge_values(self):
        """Test IPv4 with edge values (0 and 255)."""
        from src.mcp_server.models import IPToGeoRequest
        req = IPToGeoRequest(ip="0.0.0.0")
        assert req.ip == "0.0.0.0"
        
        req = IPToGeoRequest(ip="255.255.255.255")
        assert req.ip == "255.255.255.255"
    
    def test_valid_ipv6_full(self):
        """Test valid full IPv6 address."""
        from src.mcp_server.models import IPToGeoRequest
        req = IPToGeoRequest(ip="2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert req.ip == "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    
    def test_valid_ipv6_compressed(self):
        """Test valid compressed IPv6 address."""
        from src.mcp_server.models import IPToGeoRequest
        req = IPToGeoRequest(ip="::1")
        assert req.ip == "::1"
        
        req = IPToGeoRequest(ip="2001:db8::1")
        assert req.ip == "2001:db8::1"
    
    def test_invalid_ip_format(self):
        """Test that invalid format raises error."""
        from src.mcp_server.models import IPToGeoRequest
        with pytest.raises(ValidationError):
            IPToGeoRequest(ip="not-an-ip")
    
    def test_invalid_ipv4_octet_too_high(self):
        """Test IPv4 with octet > 255."""
        from src.mcp_server.models import IPToGeoRequest
        with pytest.raises(ValidationError):
            IPToGeoRequest(ip="256.1.1.1")
    
    def test_invalid_ipv4_negative_octet(self):
        """Test IPv4 with negative octet."""
        from src.mcp_server.models import IPToGeoRequest
        with pytest.raises(ValidationError):
            IPToGeoRequest(ip="-1.1.1.1")
    
    def test_invalid_empty_string(self):
        """Test that empty string raises error."""
        from src.mcp_server.models import IPToGeoRequest
        with pytest.raises(ValidationError):
            IPToGeoRequest(ip="")


class TestWeatherForecastRequest:
    """Tests for weather forecast coordinate validation."""
    
    def test_valid_coordinates(self):
        """Test valid latitude and longitude."""
        from src.mcp_server.models import WeatherForecastRequest
        req = WeatherForecastRequest(latitude=37.7749, longitude=-122.4194)
        assert req.latitude == 37.7749
        assert req.longitude == -122.4194
    
    def test_valid_edge_coordinates(self):
        """Test edge values for coordinates."""
        from src.mcp_server.models import WeatherForecastRequest
        req = WeatherForecastRequest(latitude=-90, longitude=-180)
        assert req.latitude == -90
        assert req.longitude == -180
        
        req = WeatherForecastRequest(latitude=90, longitude=180)
        assert req.latitude == 90
        assert req.longitude == 180
    
    def test_invalid_latitude_too_high(self):
        """Test latitude > 90."""
        from src.mcp_server.models import WeatherForecastRequest
        with pytest.raises(ValidationError):
            WeatherForecastRequest(latitude=91, longitude=0)
    
    def test_invalid_latitude_too_low(self):
        """Test latitude < -90."""
        from src.mcp_server.models import WeatherForecastRequest
        with pytest.raises(ValidationError):
            WeatherForecastRequest(latitude=-91, longitude=0)
    
    def test_invalid_longitude_too_high(self):
        """Test longitude > 180."""
        from src.mcp_server.models import WeatherForecastRequest
        with pytest.raises(ValidationError):
            WeatherForecastRequest(latitude=0, longitude=181)
    
    def test_invalid_longitude_too_low(self):
        """Test longitude < -180."""
        from src.mcp_server.models import WeatherForecastRequest
        with pytest.raises(ValidationError):
            WeatherForecastRequest(latitude=0, longitude=-181)


class TestWeatherDescriptions:
    """Tests for WMO weather code descriptions."""
    
    def test_known_weather_codes(self):
        """Test that known codes return correct descriptions."""
        from src.mcp_server.models import get_weather_description
        
        assert get_weather_description(0) == "Clear sky"
        assert get_weather_description(3) == "Overcast"
        assert get_weather_description(61) == "Slight rain"
        assert get_weather_description(95) == "Thunderstorm"
    
    def test_unknown_weather_code(self):
        """Test unknown code returns fallback."""
        from src.mcp_server.models import get_weather_description
        
        result = get_weather_description(999)
        assert "Unknown" in result
    
    def test_none_weather_code(self):
        """Test None code returns Unknown."""
        from src.mcp_server.models import get_weather_description
        
        assert get_weather_description(None) == "Unknown"


class TestToolCallResponse:
    """Tests for tool call response model."""
    
    def test_success_response(self):
        """Test successful response structure."""
        from src.mcp_server.models import ToolCallResponse
        
        resp = ToolCallResponse(success=True, result={"ip": "8.8.8.8"})
        assert resp.success is True
        assert resp.result == {"ip": "8.8.8.8"}
        assert resp.error is None
    
    def test_error_response(self):
        """Test error response structure."""
        from src.mcp_server.models import ToolCallResponse
        
        resp = ToolCallResponse(success=False, error="Connection failed")
        assert resp.success is False
        assert resp.result is None
        assert resp.error == "Connection failed"
