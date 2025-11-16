"""
Tests for MCP server implementation.
"""

import pytest
from unittest.mock import patch
from hitoshura25_gemini_workflow_bridge.server import mcp
from hitoshura25_gemini_workflow_bridge.server import analyze_codebase_with_gemini
from hitoshura25_gemini_workflow_bridge.server import ask_gemini


@pytest.mark.asyncio
async def test_server_initialization():
    """Test that the MCP server initializes correctly."""
    assert mcp.name == "hitoshura25_gemini_workflow_bridge"

    # Check that tools are registered (9 core tools + 2 legacy = 11 total)
    tools = await mcp.list_tools()
    assert len(tools) == 11

    tool_names = [tool.name for tool in tools]

    # Tier 1: Fact Extraction Tools
    assert "query_codebase_tool" in tool_names
    assert "find_code_by_intent_tool" in tool_names
    assert "trace_feature_tool" in tool_names
    assert "list_error_patterns_tool" in tool_names

    # Tier 2: Validation Tools
    assert "validate_against_codebase_tool" in tool_names
    assert "check_consistency_tool" in tool_names

    # Tier 3: Workflow Automation Tools
    assert "generate_feature_workflow_tool" in tool_names
    assert "generate_slash_command_tool" in tool_names
    assert "setup_workflows_tool" in tool_names

    # Legacy Tools (maintained for backward compatibility)
    assert "analyze_codebase_with_gemini" in tool_names
    assert "ask_gemini" in tool_names


@pytest.mark.asyncio
@patch('hitoshura25_gemini_workflow_bridge.server.generator.analyze_codebase_with_gemini')
async def test_analyze_codebase_with_gemini_function(mock_analyze):
    """Test analyze_codebase_with_gemini tool function."""
    # Mock the generator function to return a test result
    mock_analyze.return_value = {
        "analysis": "test analysis",
        "cached_context_id": "ctx_test123"
    }

    # Test the function directly
    result = await analyze_codebase_with_gemini(
        focus_description="test_value",
        directories="test",
        file_patterns="test",
        exclude_patterns="test"
    )

    # Result should be a string (FastMCP tools return strings)
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain dict representation
    assert "analysis" in result


@pytest.mark.asyncio
@patch('hitoshura25_gemini_workflow_bridge.server.generator.ask_gemini')
async def test_ask_gemini_function(mock_ask):
    """Test ask_gemini tool function."""
    # Mock the generator function
    mock_ask.return_value = {
        "response": "Test answer",
        "context_used": False
    }

    # Test the function directly
    result = await ask_gemini(
        prompt="test_value",
        include_codebase_context=True,
        temperature=42
    )

    # Result should be a string (FastMCP tools return strings)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_all_tools_have_descriptions():
    """Test that all tools have proper descriptions."""
    tools = await mcp.list_tools()

    for tool in tools:
        assert hasattr(tool, 'description')
        assert tool.description
        assert len(tool.description) > 0


@pytest.mark.asyncio
async def test_tool_schemas():
    """Test that all tools have proper input schemas."""
    tools = await mcp.list_tools()

    for tool in tools:
        assert hasattr(tool, 'inputSchema')
        schema = tool.inputSchema

        # Check schema structure
        assert 'type' in schema
        assert schema['type'] == 'object'
        assert 'properties' in schema


@pytest.mark.asyncio
async def test_resource_handlers():
    """Test that resource handlers are registered."""
    # Note: FastMCP resource testing might require specific setup
    # This is a basic test that the server has resources configured
    assert hasattr(mcp, '_resource_manager') or hasattr(mcp, '_resources')
