"""MCP resource handlers for workflow artifacts"""
from pathlib import Path
from typing import Dict, Any, List
import os


class WorkflowResources:
    """Manage workflow resources (specs, reviews, context)"""

    def __init__(self):
        self.specs_dir = Path(os.getenv("DEFAULT_SPEC_DIR", "./specs"))
        self.reviews_dir = Path(os.getenv("DEFAULT_REVIEW_DIR", "./reviews"))
        self.context_dir = Path(os.getenv("DEFAULT_CONTEXT_DIR", "./.workflow-context"))

        # Ensure directories exist
        self.specs_dir.mkdir(exist_ok=True)
        self.reviews_dir.mkdir(exist_ok=True)
        self.context_dir.mkdir(exist_ok=True)

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
