"""
Tests for MCP server implementation.
"""

import pytest
from unittest.mock import patch
from hitoshura25_gemini_workflow_bridge.server import mcp
from hitoshura25_gemini_workflow_bridge.server import analyze_codebase_with_gemini
from hitoshura25_gemini_workflow_bridge.server import create_specification_with_gemini
from hitoshura25_gemini_workflow_bridge.server import review_code_with_gemini
from hitoshura25_gemini_workflow_bridge.server import generate_documentation_with_gemini
from hitoshura25_gemini_workflow_bridge.server import ask_gemini


@pytest.mark.asyncio
async def test_server_initialization():
    """Test that the MCP server initializes correctly."""
    assert mcp.name == "hitoshura25_gemini_workflow_bridge"

    # Check that tools are registered
    tools = await mcp.list_tools()
    assert len(tools) == 5

    tool_names = [tool.name for tool in tools]

    assert "analyze_codebase_with_gemini" in tool_names
    assert "create_specification_with_gemini" in tool_names
    assert "review_code_with_gemini" in tool_names
    assert "generate_documentation_with_gemini" in tool_names
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
@patch('hitoshura25_gemini_workflow_bridge.server.generator.create_specification_with_gemini')
async def test_create_specification_with_gemini_function(mock_create_spec):
    """Test create_specification_with_gemini tool function."""
    # Mock the generator function
    mock_create_spec.return_value = {
        "spec_path": "./specs/test-spec.md",
        "spec_content": "# Test Spec"
    }

    # Test the function directly
    result = await create_specification_with_gemini(
        feature_description="test_value",
        context_id="test_value",
        spec_template="test_value",
        output_path="test_value"
    )

    # Result should be a string (FastMCP tools return strings)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
@patch('hitoshura25_gemini_workflow_bridge.server.generator.review_code_with_gemini')
async def test_review_code_with_gemini_function(mock_review):
    """Test review_code_with_gemini tool function."""
    # Mock the generator function
    mock_review.return_value = {
        "review_path": "./reviews/test-review.md",
        "has_blocking_issues": False
    }

    # Test the function directly
    result = await review_code_with_gemini(
        files="test",
        review_focus="test",
        spec_path="test_value",
        output_path="test_value"
    )

    # Result should be a string (FastMCP tools return strings)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
@patch('hitoshura25_gemini_workflow_bridge.server.generator.generate_documentation_with_gemini')
async def test_generate_documentation_with_gemini_function(mock_gen_doc):
    """Test generate_documentation_with_gemini tool function."""
    # Mock the generator function
    mock_gen_doc.return_value = {
        "doc_path": "./docs/test-doc.md",
        "word_count": 100
    }

    # Test the function directly
    result = await generate_documentation_with_gemini(
        documentation_type="test_value",
        scope="test_value",
        output_path="test_value",
        include_examples=True
    )

    # Result should be a string (FastMCP tools return strings)
    assert isinstance(result, str)
    assert len(result) > 0


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
        context_id="test_value",
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
