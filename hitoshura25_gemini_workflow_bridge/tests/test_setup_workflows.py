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
        assert "Unknown workflow type" in result["skipped"][0]["reason"]
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_setup_workflows_path_traversal_protection():
    """Test that setup_workflows protects against path traversal."""
    result = await setup_workflows(
        workflows=["spec-only"],
        output_dir="../../../etc"
    )

    # Should detect path traversal attempt
    assert result["success"] is False
    assert "path traversal" in result["message"]


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
        result = await setup_workflows(
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
