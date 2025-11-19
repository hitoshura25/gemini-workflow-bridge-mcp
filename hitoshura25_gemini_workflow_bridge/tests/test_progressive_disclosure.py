"""
Tests for progressive tool disclosure.
"""

import json

import pytest

from hitoshura25_gemini_workflow_bridge.server import discover_tools, get_tool_schema
from hitoshura25_gemini_workflow_bridge.tool_registry import (
    TOOL_REGISTRY,
    ToolCategory,
    get_all_tool_names,
    get_tool_metadata,
    get_tools_by_category,
    search_tools_by_keyword,
)


class TestToolRegistry:
    """Tests for tool registry and metadata"""

    def test_tool_registry_not_empty(self):
        """Test tool registry contains tools"""
        assert len(TOOL_REGISTRY) > 0

    def test_get_tool_metadata(self):
        """Test getting metadata for specific tool"""
        metadata = get_tool_metadata("query_codebase_tool")
        assert metadata is not None
        assert metadata["name"] == "query_codebase_tool"
        assert metadata["category"] == ToolCategory.FACT_EXTRACTION
        assert "keywords" in metadata
        assert "use_cases" in metadata

    def test_get_tool_metadata_not_found(self):
        """Test getting metadata for non-existent tool"""
        metadata = get_tool_metadata("nonexistent_tool")
        assert metadata is None

    def test_get_all_tool_names(self):
        """Test getting all tool names"""
        names = get_all_tool_names()
        assert len(names) > 0
        assert "query_codebase_tool" in names
        assert "ask_gemini" in names
        assert "setup_workflows_tool" in names

    def test_get_tools_by_category(self):
        """Test filtering tools by category"""
        fact_tools = get_tools_by_category(ToolCategory.FACT_EXTRACTION)
        assert len(fact_tools) > 0
        for tool in fact_tools:
            assert tool["category"] == ToolCategory.FACT_EXTRACTION

        validation_tools = get_tools_by_category(ToolCategory.VALIDATION)
        assert len(validation_tools) > 0
        for tool in validation_tools:
            assert tool["category"] == ToolCategory.VALIDATION

    def test_search_tools_by_keyword(self):
        """Test searching tools by keyword"""
        # Search by keyword
        results = search_tools_by_keyword("query")
        assert len(results) > 0
        assert any("query" in tool["name"].lower() for tool in results)

        # Search by description
        results = search_tools_by_keyword("validate")
        assert len(results) > 0

        # Search by use case
        results = search_tools_by_keyword("specification")
        assert len(results) > 0

    def test_search_tools_case_insensitive(self):
        """Test search is case-insensitive"""
        results_lower = search_tools_by_keyword("query")
        results_upper = search_tools_by_keyword("QUERY")
        results_mixed = search_tools_by_keyword("Query")

        assert len(results_lower) == len(results_upper)
        assert len(results_lower) == len(results_mixed)

    def test_search_tools_partial_match(self):
        """Test search supports partial matches"""
        results = search_tools_by_keyword("code")
        assert len(results) > 0
        # Should match "codebase", "code analysis", etc.

    def test_all_tools_have_required_fields(self):
        """Test all tools have required metadata fields"""
        for tool_name, metadata in TOOL_REGISTRY.items():
            assert "name" in metadata
            assert "category" in metadata
            assert "short_description" in metadata
            assert "keywords" in metadata
            assert "use_cases" in metadata
            assert "complexity" in metadata
            assert "requires_codebase" in metadata

            # Check field types
            assert isinstance(metadata["name"], str)
            assert isinstance(metadata["category"], ToolCategory)
            assert isinstance(metadata["short_description"], str)
            assert isinstance(metadata["keywords"], list)
            assert isinstance(metadata["use_cases"], list)
            assert isinstance(metadata["complexity"], str)
            assert isinstance(metadata["requires_codebase"], bool)


class TestDiscoverTools:
    """Tests for discover_tools function"""

    @pytest.mark.asyncio
    async def test_discover_all_tools_name_only(self):
        """Test discovering all tools with name_only detail level"""
        result = await discover_tools(detail_level="name_only")
        data = json.loads(result)

        assert data["detail_level"] == "name_only"
        assert data["tool_count"] > 0
        assert isinstance(data["tools"], list)
        # With name_only, tools should be strings
        assert all(isinstance(tool, str) for tool in data["tools"])

    @pytest.mark.asyncio
    async def test_discover_all_tools_with_description(self):
        """Test discovering all tools with descriptions"""
        result = await discover_tools(detail_level="with_description")
        data = json.loads(result)

        assert data["detail_level"] == "with_description"
        assert data["tool_count"] > 0
        # With with_description, tools should be dicts
        assert all(isinstance(tool, dict) for tool in data["tools"])
        for tool in data["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "complexity" in tool

    @pytest.mark.asyncio
    async def test_discover_all_tools_with_use_cases(self):
        """Test discovering all tools with use cases"""
        result = await discover_tools(detail_level="with_use_cases")
        data = json.loads(result)

        assert data["detail_level"] == "with_use_cases"
        for tool in data["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "use_cases" in tool
            assert "requires_codebase" in tool
            assert isinstance(tool["use_cases"], list)

    @pytest.mark.asyncio
    async def test_discover_all_tools_full(self):
        """Test discovering all tools with full detail"""
        result = await discover_tools(detail_level="full")
        data = json.loads(result)

        assert data["detail_level"] == "full"
        for tool in data["tools"]:
            assert "name" in tool
            assert "category" in tool
            assert "keywords" in tool
            assert "use_cases" in tool

    @pytest.mark.asyncio
    async def test_discover_by_query(self):
        """Test discovering tools by search query"""
        result = await discover_tools(query="validate")
        data = json.loads(result)

        assert data["query"] == "validate"
        assert data["tool_count"] > 0
        # Should find validation-related tools
        tool_names = [tool["name"] for tool in data["tools"]]
        assert any("validate" in name.lower() for name in tool_names)

    @pytest.mark.asyncio
    async def test_discover_by_category(self):
        """Test discovering tools by category"""
        result = await discover_tools(category="validation")
        data = json.loads(result)

        assert data["category"] == "validation"
        assert data["tool_count"] > 0
        # All tools should be validation tools
        # (Can't verify category directly since it depends on detail_level)

    @pytest.mark.asyncio
    async def test_discover_invalid_category(self):
        """Test discovering tools with invalid category"""
        result = await discover_tools(category="invalid_category")
        data = json.loads(result)

        assert "error" in data
        assert "valid_categories" in data

    @pytest.mark.asyncio
    async def test_discover_invalid_detail_level(self):
        """Test discovering tools with invalid detail level"""
        result = await discover_tools(detail_level="invalid")
        data = json.loads(result)

        assert "error" in data
        assert "valid_levels" in data

    @pytest.mark.asyncio
    async def test_discover_codebase_analysis_tools(self):
        """Test discovering tools for codebase analysis"""
        result = await discover_tools(query="codebase")
        data = json.loads(result)

        assert data["tool_count"] > 0
        # Should find tools related to codebase (search is partial match)

    @pytest.mark.asyncio
    async def test_discover_workflow_tools(self):
        """Test discovering workflow setup tools"""
        result = await discover_tools(category="workflow")
        data = json.loads(result)

        assert data["tool_count"] > 0
        tool_names = [tool["name"] for tool in data["tools"]]
        assert "setup_workflows_tool" in tool_names


class TestGetToolSchema:
    """Tests for get_tool_schema function"""

    @pytest.mark.asyncio
    async def test_get_schema_for_valid_tool(self):
        """Test getting schema for valid tool"""
        result = await get_tool_schema("query_codebase_tool")
        data = json.loads(result)

        assert data["name"] == "query_codebase_tool"
        assert "category" in data
        assert "description" in data
        assert "complexity" in data
        assert "requires_codebase" in data
        assert "use_cases" in data
        assert "keywords" in data
        assert "usage_hint" in data

    @pytest.mark.asyncio
    async def test_get_schema_for_invalid_tool(self):
        """Test getting schema for non-existent tool"""
        result = await get_tool_schema("nonexistent_tool")
        data = json.loads(result)

        assert "error" in data
        assert "available_tools" in data
        assert "hint" in data

    @pytest.mark.asyncio
    async def test_get_schema_includes_usage_hint(self):
        """Test schema includes usage hint"""
        result = await get_tool_schema("ask_gemini")
        data = json.loads(result)

        assert "usage_hint" in data
        assert "ask_gemini" in data["usage_hint"]

    @pytest.mark.asyncio
    async def test_get_schema_for_all_tools(self):
        """Test getting schema for all registered tools"""
        for tool_name in get_all_tool_names():
            result = await get_tool_schema(tool_name)
            data = json.loads(result)

            # Should not have error
            assert "error" not in data
            assert data["name"] == tool_name


class TestProgressiveDisclosureWorkflow:
    """Integration tests for progressive disclosure workflow"""

    @pytest.mark.asyncio
    async def test_discovery_workflow(self):
        """Test complete discovery workflow: search -> get schema -> use tool"""
        # Step 1: Discover tools for validation
        discover_result = await discover_tools(
            query="validate specification",
            detail_level="with_description"
        )
        discover_data = json.loads(discover_result)

        assert discover_data["tool_count"] > 0

        # Step 2: Get detailed schema for first tool
        first_tool_name = discover_data["tools"][0]["name"]
        schema_result = await get_tool_schema(first_tool_name)
        schema_data = json.loads(schema_result)

        assert schema_data["name"] == first_tool_name
        assert "usage_hint" in schema_data

        # Step 3: Tool would be used (not tested here since it requires actual implementation)

    @pytest.mark.asyncio
    async def test_category_based_discovery(self):
        """Test discovering tools by category then getting details"""
        # Discover all fact extraction tools
        result = await discover_tools(
            category="fact_extraction",
            detail_level="name_only"
        )
        data = json.loads(result)

        assert data["tool_count"] > 0
        assert isinstance(data["tools"], list)

        # Get schema for each tool
        for tool_name in data["tools"]:
            schema_result = await get_tool_schema(tool_name)
            schema_data = json.loads(schema_result)
            assert schema_data["category"] == "fact_extraction"

    @pytest.mark.asyncio
    async def test_progressive_detail_levels(self):
        """Test progressive loading of tool details"""
        # Start with minimal info
        result1 = await discover_tools(query="codebase", detail_level="name_only")
        data1 = json.loads(result1)

        # Get more details
        result2 = await discover_tools(query="codebase", detail_level="with_description")
        data2 = json.loads(result2)

        # Get full details
        result3 = await discover_tools(query="codebase", detail_level="full")
        data3 = json.loads(result3)

        # Each level should have same tools but increasing detail
        assert data1["tool_count"] == data2["tool_count"] == data3["tool_count"]
