"""
Tests for setup_workflows tool.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from hitoshura25_gemini_workflow_bridge.tools.setup_workflows import setup_workflows


@pytest.mark.asyncio
async def test_setup_workflows_default():
    """Test setup_workflows with default parameters (spec-only)."""
    temp_dir = tempfile.mkdtemp()
    try:
        result = await setup_workflows(output_dir=temp_dir)

        assert result["success"] is True
        assert len(result["workflows_created"]) == 1
        assert result["workflows_created"][0]["name"] == "spec-only"

        # Verify files were created
        workflow_file = Path(temp_dir) / ".claude" / "workflows" / "spec-only.md"
        command_file = Path(temp_dir) / ".claude" / "commands" / "spec-only.md"
        assert workflow_file.exists()
        assert command_file.exists()

        # Verify content is not empty
        assert len(workflow_file.read_text()) > 0
        assert len(command_file.read_text()) > 0
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_setup_workflows_all():
    """Test setup_workflows with 'all' workflows."""
    temp_dir = tempfile.mkdtemp()
    try:
        result = await setup_workflows(workflows=["all"], output_dir=temp_dir)

        assert result["success"] is True
        assert len(result["workflows_created"]) == 4

        # Verify all workflow types were created
        workflow_names = [w["name"] for w in result["workflows_created"]]
        assert "spec-only" in workflow_names
        assert "feature" in workflow_names
        assert "refactor" in workflow_names
        assert "review" in workflow_names

        # Verify all files exist
        for workflow_name in ["spec-only", "feature", "refactor", "review"]:
            workflow_file = Path(temp_dir) / ".claude" / "workflows" / f"{workflow_name}.md"
            command_file = Path(temp_dir) / ".claude" / "commands" / f"{workflow_name}.md"
            assert workflow_file.exists()
            assert command_file.exists()
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_setup_workflows_without_commands():
    """Test setup_workflows without creating command files."""
    temp_dir = tempfile.mkdtemp()
    try:
        result = await setup_workflows(
            workflows=["spec-only"],
            output_dir=temp_dir,
            include_commands=False
        )

        assert result["success"] is True
        assert len(result["workflows_created"]) == 1

        # Verify workflow file exists but command file does not
        workflow_file = Path(temp_dir) / ".claude" / "workflows" / "spec-only.md"
        command_file = Path(temp_dir) / ".claude" / "commands" / "spec-only.md"
        assert workflow_file.exists()
        assert not command_file.exists()
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_setup_workflows_overwrite_protection():
    """Test that setup_workflows respects overwrite protection."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create workflow first time
        result1 = await setup_workflows(output_dir=temp_dir)
        assert result1["success"] is True

        # Try to create again without overwrite
        result2 = await setup_workflows(output_dir=temp_dir)
        assert result2["success"] is False  # Should fail because nothing was created
        assert len(result2["skipped"]) == 1
        assert "already exists" in result2["skipped"][0]["status"]

        # Create with overwrite should succeed
        result3 = await setup_workflows(output_dir=temp_dir, overwrite=True)
        assert result3["success"] is True
        assert len(result3["workflows_created"]) == 1
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_setup_workflows_invalid_workflow():
    """Test setup_workflows with invalid workflow name."""
    temp_dir = tempfile.mkdtemp()
    try:
        result = await setup_workflows(
            workflows=["invalid-workflow"],
            output_dir=temp_dir
        )

        assert result["success"] is False
        assert len(result["skipped"]) == 1

        # Verify consistent structure for skipped items
        skipped_item = result["skipped"][0]
        assert "name" in skipped_item
        assert "workflow_path" in skipped_item
        assert "command_path" in skipped_item
        assert "reason" in skipped_item
        assert "Unknown workflow type" in skipped_item["reason"]
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_setup_workflows_resolves_paths():
    """Test that setup_workflows resolves paths correctly."""
    # Path.resolve() will normalize any path traversal attempts
    # This test verifies the function handles path resolution without errors
    temp_dir = tempfile.mkdtemp()
    try:
        # Use a path with .. that resolves to temp_dir
        relative_path = str(Path(temp_dir) / "subdir" / "..")
        result = await setup_workflows(
            workflows=["spec-only"],
            output_dir=relative_path
        )

        # Should succeed - path is resolved to temp_dir
        assert result["success"] is True

        # Verify files were created in the resolved location
        workflow_file = Path(temp_dir) / ".claude" / "workflows" / "spec-only.md"
        assert workflow_file.exists()
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_setup_workflows_multiple_workflows():
    """Test setup_workflows with specific multiple workflows."""
    temp_dir = tempfile.mkdtemp()
    try:
        result = await setup_workflows(
            workflows=["spec-only", "feature"],
            output_dir=temp_dir
        )

        assert result["success"] is True
        assert len(result["workflows_created"]) == 2

        workflow_names = [w["name"] for w in result["workflows_created"]]
        assert "spec-only" in workflow_names
        assert "feature" in workflow_names
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_workflow_content_validity():
    """Test that generated workflow files contain expected content."""
    temp_dir = tempfile.mkdtemp()
    try:
        await setup_workflows(
            workflows=["spec-only"],
            output_dir=temp_dir
        )

        workflow_file = Path(temp_dir) / ".claude" / "workflows" / "spec-only.md"
        command_file = Path(temp_dir) / ".claude" / "commands" / "spec-only.md"

        workflow_content = workflow_file.read_text()
        command_content = command_file.read_text()

        # Verify workflow content
        assert "# Specification-Only Workflow" in workflow_content
        assert "## Purpose" in workflow_content
        assert "## Steps" in workflow_content
        assert "query_codebase_tool" in workflow_content

        # Verify command content
        assert "/spec-only" in command_content
        assert "## Usage" in command_content
        assert "## Description" in command_content
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_command_counting_logic():
    """Test that command counting correctly reflects only newly created commands."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create spec-only workflow and command
        result1 = await setup_workflows(
            workflows=["spec-only"],
            output_dir=temp_dir
        )
        assert "Successfully set up 1 workflow(s) and 1 command(s)" in result1["message"]

        # Create feature and refactor workflows, but spec-only command already exists
        result2 = await setup_workflows(
            workflows=["spec-only", "feature", "refactor"],
            output_dir=temp_dir
        )

        # Should report:
        # - 2 workflows created (feature, refactor)
        # - 2 commands created (feature, refactor)
        # - 1 skipped (spec-only already exists)
        assert len(result2["workflows_created"]) == 2
        assert len(result2["skipped"]) == 1
        assert "Successfully set up 2 workflow(s) and 2 command(s)" in result2["message"]

        # Verify the skipped one was spec-only
        assert result2["skipped"][0]["name"] == "spec-only"

    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_duplicate_workflow_handling():
    """Test that duplicate workflow names are handled correctly."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Request spec-only twice explicitly
        result = await setup_workflows(
            workflows=["spec-only", "feature", "spec-only"],
            output_dir=temp_dir
        )

        # Should create 2 workflows (spec-only and feature), not 3
        assert result["success"] is True
        assert len(result["workflows_created"]) == 2

        workflow_names = [w["name"] for w in result["workflows_created"]]
        assert "spec-only" in workflow_names
        assert "feature" in workflow_names
        # Verify spec-only appears only once
        assert workflow_names.count("spec-only") == 1

    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_all_with_duplicates():
    """Test that 'all' with other workflows removes duplicates."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Request "all" along with spec-only explicitly
        result = await setup_workflows(
            workflows=["spec-only", "all"],
            output_dir=temp_dir
        )

        # Should create all 4 workflows without duplication
        assert result["success"] is True
        assert len(result["workflows_created"]) == 4

        workflow_names = [w["name"] for w in result["workflows_created"]]
        assert "spec-only" in workflow_names
        assert "feature" in workflow_names
        assert "refactor" in workflow_names
        assert "review" in workflow_names
        # Verify no duplicates
        assert len(workflow_names) == len(set(workflow_names))

    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_environment_variable_support():
    """Test that DEFAULT_WORKFLOW_DIR and DEFAULT_COMMAND_DIR are respected."""
    import os
    temp_dir = tempfile.mkdtemp()
    try:
        # Set custom environment variables
        old_workflow_dir = os.environ.get("DEFAULT_WORKFLOW_DIR")
        old_command_dir = os.environ.get("DEFAULT_COMMAND_DIR")

        os.environ["DEFAULT_WORKFLOW_DIR"] = "custom/workflows"
        os.environ["DEFAULT_COMMAND_DIR"] = "custom/commands"

        result = await setup_workflows(
            workflows=["spec-only"],
            output_dir=temp_dir
        )

        # Verify files were created in custom directories
        workflow_file = Path(temp_dir) / "custom" / "workflows" / "spec-only.md"
        command_file = Path(temp_dir) / "custom" / "commands" / "spec-only.md"

        assert workflow_file.exists(), f"Expected workflow at {workflow_file}"
        assert command_file.exists(), f"Expected command at {command_file}"

        # Restore original environment
        if old_workflow_dir:
            os.environ["DEFAULT_WORKFLOW_DIR"] = old_workflow_dir
        else:
            os.environ.pop("DEFAULT_WORKFLOW_DIR", None)

        if old_command_dir:
            os.environ["DEFAULT_COMMAND_DIR"] = old_command_dir
        else:
            os.environ.pop("DEFAULT_COMMAND_DIR", None)

    finally:
        shutil.rmtree(temp_dir)
