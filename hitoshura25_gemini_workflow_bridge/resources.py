"""MCP resource handlers for workflow artifacts"""
from pathlib import Path
from typing import Dict, Any, List
import os
import logging

logger = logging.getLogger(__name__)


class WorkflowResources:
    """Manage workflow resources (specs, reviews, context) and configuration"""

    def __init__(self):
        # Directory configuration
        self.specs_dir = Path(os.getenv("DEFAULT_SPEC_DIR", "./specs"))
        self.reviews_dir = Path(os.getenv("DEFAULT_REVIEW_DIR", "./reviews"))
        self.context_dir = Path(os.getenv("DEFAULT_CONTEXT_DIR", "./.workflow-context"))

        # Prefix configuration
        self.command_prefix = os.getenv("GEMINI_COMMAND_PREFIX", "gemini-")
        self.workflow_prefix = os.getenv("GEMINI_WORKFLOW_PREFIX", "gemini-")

        # Validate prefixes
        self._validate_prefix(self.command_prefix, "GEMINI_COMMAND_PREFIX")
        self._validate_prefix(self.workflow_prefix, "GEMINI_WORKFLOW_PREFIX")

        # Ensure directories exist
        self.specs_dir.mkdir(exist_ok=True)
        self.reviews_dir.mkdir(exist_ok=True)
        self.context_dir.mkdir(exist_ok=True)

    @staticmethod
    def _validate_prefix(prefix: str, env_var: str) -> None:
        """Validate prefix format (optional safety check)

        Rules:
        - Can be empty (disabled)
        - Should be alphanumeric with dashes/underscores
        - Should not contain spaces or special characters
        - Recommend ending with dash for readability
        """
        if prefix == "":
            # Empty prefix is allowed (disables prefixing)
            return

        # Warn about potentially problematic prefixes
        if " " in prefix:
            logger.warning(
                f"{env_var}='{prefix}' contains spaces. "
                f"This may cause issues with command parsing."
            )

        if not prefix.endswith("-") and not prefix.endswith("_"):
            logger.info(
                f"{env_var}='{prefix}' does not end with '-' or '_'. "
                f"Consider adding a separator for better readability "
                f"(e.g., 'gemini-' instead of 'gemini')."
            )

    def list_resources(self) -> List[str]:
        """List all available resources"""
        resources = []

        # Specs
        for spec_file in self.specs_dir.glob("*.md"):
            uri = f"workflow://specs/{spec_file.stem}"
            resources.append(uri)

        # Reviews
        for review_file in self.reviews_dir.glob("*.md"):
            uri = f"workflow://reviews/{review_file.stem}"
            resources.append(uri)

        # Cached contexts
        for context_file in self.context_dir.glob("*.json"):
            uri = f"workflow://context/{context_file.stem}"
            resources.append(uri)

        return resources

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource by URI"""
        if uri.startswith("workflow://specs/"):
            name = uri.replace("workflow://specs/", "")
            file_path = self.specs_dir / f"{name}.md"

            if file_path.exists():
                return {
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": file_path.read_text()
                }

        elif uri.startswith("workflow://reviews/"):
            name = uri.replace("workflow://reviews/", "")
            file_path = self.reviews_dir / f"{name}.md"

            if file_path.exists():
                return {
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": file_path.read_text()
                }

        elif uri.startswith("workflow://context/"):
            name = uri.replace("workflow://context/", "")
            file_path = self.context_dir / f"{name}.json"

            if file_path.exists():
                return {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": file_path.read_text()
                }

        raise ValueError(f"Resource not found: {uri}")


# Create singleton instance for import
workflow_resources = WorkflowResources()
