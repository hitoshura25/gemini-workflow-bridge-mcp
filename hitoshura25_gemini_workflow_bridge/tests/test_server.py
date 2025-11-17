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


@pytest.mark.asyncio
@patch('hitoshura25_gemini_workflow_bridge.tools.validate_spec.GeminiClient')
async def test_validate_against_codebase_response_format(mock_gemini_client_class):
    """Test that validate_against_codebase returns the expected response format with new fields."""
    from hitoshura25_gemini_workflow_bridge.tools.validate_spec import validate_against_codebase
    from unittest.mock import AsyncMock
    import json

    # Mock the Gemini response with the new format
    mock_gemini_response = {
        "validation_result": "pass_with_warnings",
        "completeness_score": 0.7,
        "score_breakdown": {
            "what_was_checked": ["file_existence", "dependency_presence"],
            "what_passed": ["All files exist"],
            "what_failed": [],
            "what_was_skipped": ["code_pattern_analysis - no code content provided"]
        },
        "verification_limitations": {
            "available_context": ["file paths", "package.json content"],
            "missing_context": ["function signatures", "implementation code"],
            "impact_on_score": "Limited to file and dependency validation"
        },
        "issues": [],
        "missing_elements": {
            "files": [],
            "dependencies": [],
            "functions": []
        },
        "pattern_alignment": {
            "matches_existing_patterns": True,
            "conflicts": [],
            "suggestions": []
        }
    }

    # Setup mock - generate_content is async so use AsyncMock
    mock_client_instance = mock_gemini_client_class.return_value
    mock_client_instance.generate_content = AsyncMock(return_value=json.dumps(mock_gemini_response))

    # Call the function with minimal test data
    result = await validate_against_codebase(
        spec_content="# Test Spec\nSome test specification content",
        validation_checks=["missing_files", "undefined_dependencies"],
        codebase_context="# Codebase Files\n- test.py"
    )

    # Verify all expected fields are present
    assert "validation_result" in result
    assert "completeness_score" in result
    assert "score_breakdown" in result
    assert "verification_limitations" in result
    assert "issues" in result
    assert "missing_elements" in result
    assert "pattern_alignment" in result
    assert "metadata" in result

    # Verify new fields have the expected structure
    assert "what_was_checked" in result["score_breakdown"]
    assert "what_passed" in result["score_breakdown"]
    assert "what_failed" in result["score_breakdown"]
    assert "what_was_skipped" in result["score_breakdown"]

    assert "available_context" in result["verification_limitations"]
    assert "missing_context" in result["verification_limitations"]
    assert "impact_on_score" in result["verification_limitations"]

    # Verify score is the expected value
    assert result["completeness_score"] == 0.7
