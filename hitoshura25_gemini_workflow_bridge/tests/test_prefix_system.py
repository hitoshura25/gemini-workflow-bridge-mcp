"""
Tests for command prefix system.
"""

import os
from unittest.mock import patch

import pytest

from hitoshura25_gemini_workflow_bridge.resources import WorkflowResources
from hitoshura25_gemini_workflow_bridge.tools.generate_command import generate_slash_command
from hitoshura25_gemini_workflow_bridge.tools.setup_workflows import setup_workflows


class TestPrefixConfiguration:
    """Tests for prefix configuration in WorkflowResources"""

    def test_default_prefixes(self):
        """Test default prefixes are set correctly"""
        with patch.dict(os.environ, {}, clear=True):
            resources = WorkflowResources()
            assert resources.command_prefix == "gemini-"
            assert resources.workflow_prefix == "gemini-"

    def test_custom_prefixes_from_env(self):
        """Test custom prefixes from environment variables"""
        with patch.dict(os.environ, {
            "GEMINI_COMMAND_PREFIX": "custom-cmd-",
            "GEMINI_WORKFLOW_PREFIX": "custom-wf-"
        }, clear=True):
            resources = WorkflowResources()
            assert resources.command_prefix == "custom-cmd-"
            assert resources.workflow_prefix == "custom-wf-"

    def test_empty_prefix(self):
        """Test empty prefix (disabled)"""
        with patch.dict(os.environ, {
            "GEMINI_COMMAND_PREFIX": "",
            "GEMINI_WORKFLOW_PREFIX": ""
        }, clear=True):
            resources = WorkflowResources()
            assert resources.command_prefix == ""
            assert resources.workflow_prefix == ""

    def test_prefix_validation_with_spaces(self):
        """Test prefix validation warns about spaces"""
        with patch.dict(os.environ, {
            "GEMINI_COMMAND_PREFIX": "test prefix"
        }, clear=True):
            # Should not raise, but will log warning
            resources = WorkflowResources()
            assert resources.command_prefix == "test prefix"

    def test_prefix_validation_without_separator(self):
        """Test prefix validation suggests separator"""
        with patch.dict(os.environ, {
            "GEMINI_COMMAND_PREFIX": "gemini"
        }, clear=True):
            # Should not raise, but will log info
            resources = WorkflowResources()
            assert resources.command_prefix == "gemini"


class TestSetupWorkflowsWithPrefix:
    """Tests for setup_workflows with prefix support"""

    @pytest.mark.asyncio
    async def test_setup_with_default_prefix(self, tmp_path):
        """Test workflow setup uses default prefix"""
        with patch.dict(os.environ, {
            "GEMINI_COMMAND_PREFIX": "gemini-",
            "GEMINI_WORKFLOW_PREFIX": "gemini-"
        }, clear=True):
            result = await setup_workflows(
                workflows=["spec-only"],
                output_dir=str(tmp_path),
                overwrite=True
            )

            assert result["success"] is True
            assert len(result["workflows_created"]) == 1

            workflow_info = result["workflows_created"][0]
            assert workflow_info["name"] == "spec-only"
            assert workflow_info["prefixed_name"] == "gemini-spec-only"
            assert "gemini-spec-only.md" in workflow_info["workflow_path"]
            assert "gemini-spec-only.md" in workflow_info["command_path"]

            # Check files actually exist
            workflow_file = tmp_path / ".claude" / "workflows" / "gemini-spec-only.md"
            command_file = tmp_path / ".claude" / "commands" / "gemini-spec-only.md"
            assert workflow_file.exists()
            assert command_file.exists()

    @pytest.mark.asyncio
    async def test_setup_with_custom_prefix(self, tmp_path):
        """Test workflow setup with custom prefix"""
        result = await setup_workflows(
            workflows=["feature"],
            output_dir=str(tmp_path),
            overwrite=True,
            command_prefix="custom-",
            workflow_prefix="wf-"
        )

        assert result["success"] is True
        workflow_info = result["workflows_created"][0]
        assert workflow_info["prefixed_name"] == "wf-feature"
        assert "wf-feature.md" in workflow_info["workflow_path"]
        assert "custom-feature.md" in workflow_info["command_path"]

        # Check files exist with custom prefix
        workflow_file = tmp_path / ".claude" / "workflows" / "wf-feature.md"
        command_file = tmp_path / ".claude" / "commands" / "custom-feature.md"
        assert workflow_file.exists()
        assert command_file.exists()

    @pytest.mark.asyncio
    async def test_setup_with_empty_prefix(self, tmp_path):
        """Test workflow setup with empty prefix (disabled)"""
        result = await setup_workflows(
            workflows=["refactor"],
            output_dir=str(tmp_path),
            overwrite=True,
            command_prefix="",
            workflow_prefix=""
        )

        assert result["success"] is True
        workflow_info = result["workflows_created"][0]
        assert workflow_info["prefixed_name"] == "refactor"
        assert "refactor.md" in workflow_info["workflow_path"]

        # Check files exist without prefix
        workflow_file = tmp_path / ".claude" / "workflows" / "refactor.md"
        command_file = tmp_path / ".claude" / "commands" / "refactor.md"
        assert workflow_file.exists()
        assert command_file.exists()

    @pytest.mark.asyncio
    async def test_command_content_uses_prefix(self, tmp_path):
        """Test command file content uses prefixed command name"""
        await setup_workflows(
            workflows=["spec-only"],
            output_dir=str(tmp_path),
            overwrite=True,
            command_prefix="test-",
            workflow_prefix="test-"
        )

        command_file = tmp_path / ".claude" / "commands" / "test-spec-only.md"
        content = command_file.read_text()

        # Check that command content uses prefixed name
        assert "/test-spec-only" in content
        # Ensure /spec-only doesn't appear standalone (only as part of /test-spec-only)
        # Check for /spec-only with word boundaries (space, newline, etc.)
        import re
        standalone_unprefixed = re.search(r'(?<![a-z-])/spec-only(?![a-z-])', content)
        assert standalone_unprefixed is None, \
            "Found standalone '/spec-only' in content, prefix substitution incomplete"

    @pytest.mark.asyncio
    async def test_multiple_workflows_with_prefix(self, tmp_path):
        """Test setting up multiple workflows with prefix"""
        result = await setup_workflows(
            workflows=["spec-only", "feature", "refactor"],
            output_dir=str(tmp_path),
            overwrite=True,
            command_prefix="gw-",
            workflow_prefix="gw-"
        )

        assert result["success"] is True
        assert len(result["workflows_created"]) == 3

        for workflow_info in result["workflows_created"]:
            assert workflow_info["prefixed_name"].startswith("gw-")


class TestGenerateCommandWithPrefix:
    """Tests for generate_slash_command with prefix support"""

    @pytest.mark.asyncio
    async def test_generate_with_default_prefix(self, tmp_path):
        """Test command generation uses default prefix"""
        with patch.dict(os.environ, {
            "GEMINI_COMMAND_PREFIX": "gemini-"
        }, clear=True):
            result = await generate_slash_command(
                command_name="test-feature",
                workflow_type="feature",
                description="Test feature command",
                save_to=str(tmp_path / "test.md")
            )

            assert result["prefixed_name"] == "gemini-test-feature"
            assert result["base_name"] == "test-feature"
            assert "/gemini-test-feature" in result["usage_example"]
            assert "/gemini-test-feature" in result["command_content"]

    @pytest.mark.asyncio
    async def test_generate_with_custom_prefix(self, tmp_path):
        """Test command generation with custom prefix"""
        result = await generate_slash_command(
            command_name="deploy",
            workflow_type="custom",
            description="Deploy application",
            steps=["Build", "Test", "Deploy"],
            save_to=str(tmp_path / "deploy.md"),
            prefix="prod-"
        )

        assert result["prefixed_name"] == "prod-deploy"
        assert result["base_name"] == "deploy"
        assert "/prod-deploy" in result["usage_example"]

        # When save_to is provided, it uses that exact path
        assert result["command_path"] == str(tmp_path / "deploy.md")

    @pytest.mark.asyncio
    async def test_generate_with_empty_prefix(self, tmp_path):
        """Test command generation with empty prefix"""
        result = await generate_slash_command(
            command_name="review",
            workflow_type="review",
            description="Review code",
            save_to=str(tmp_path / "review.md"),
            prefix=""
        )

        assert result["prefixed_name"] == "review"
        assert result["base_name"] == "review"
        assert "/review" in result["usage_example"]

    @pytest.mark.asyncio
    async def test_generate_command_content_prefix(self, tmp_path):
        """Test generated command content uses prefix throughout"""
        result = await generate_slash_command(
            command_name="analyze",
            workflow_type="custom",
            description="Analyze codebase",
            steps=["Step 1", "Step 2"],
            save_to=str(tmp_path / "analyze.md"),
            prefix="custom-"
        )

        content = result["command_content"]

        # Check prefix is used in command header
        assert "# /custom-analyze" in content
        # Check prefix is not mixed with non-prefixed version
        assert content.count("/custom-analyze") > 0

    @pytest.mark.asyncio
    async def test_generate_default_save_path_uses_prefix(self, tmp_path):
        """Test default save path uses prefixed filename"""
        with patch.dict(os.environ, {
            "DEFAULT_COMMAND_DIR": str(tmp_path / ".claude/commands"),
            "GEMINI_COMMAND_PREFIX": "mcp-"
        }, clear=True):
            result = await generate_slash_command(
                command_name="build",
                workflow_type="custom",
                description="Build project",
                steps=["Compile", "Link"],
                prefix="mcp-"
            )

            assert "mcp-build.md" in result["command_path"]
