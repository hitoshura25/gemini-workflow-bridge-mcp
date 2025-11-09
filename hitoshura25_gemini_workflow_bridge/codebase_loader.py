"""Codebase loading utilities"""
from pathlib import Path
from typing import List, Optional, Dict
import pathspec
import git

class CodebaseLoader:
    """Load and prepare codebase for analysis"""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()
        self.gitignore_spec = self._load_gitignore()

    def _load_gitignore(self) -> Optional[pathspec.PathSpec]:
        """Load .gitignore patterns"""
        gitignore_path = self.root_dir / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path) as f:
                patterns = f.read().splitlines()
            return pathspec.PathSpec.from_lines('gitwildmatch', patterns)
        return None

    def load_files(
        self,
        file_patterns: List[str] = ["*.py", "*.js", "*.ts", "*.java"],
        exclude_patterns: List[str] = [],
        directories: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Load files matching patterns"""
        files_content = {}

        search_dirs = [Path(d) for d in directories] if directories else [self.root_dir]

        for search_dir in search_dirs:
            for pattern in file_patterns:
                for file_path in search_dir.rglob(pattern):
                    # Skip if matches exclude patterns
                    if self._should_exclude(file_path, exclude_patterns):
                        continue

                    # Skip if in .gitignore
                    if self.gitignore_spec and self.gitignore_spec.match_file(
                        str(file_path.relative_to(self.root_dir))
                    ):
                        continue

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            relative_path = file_path.relative_to(self.root_dir)
                            files_content[str(relative_path)] = f.read()
                    except (UnicodeDecodeError, PermissionError):
                        # Skip binary or inaccessible files
                        continue

        return files_content

    def _should_exclude(self, file_path: Path, exclude_patterns: List[str]) -> bool:
        """Check if file should be excluded"""
        for pattern in exclude_patterns:
            if pattern in str(file_path):
                return True
        return False

    def get_project_structure(self) -> str:
        """Get ASCII tree of project structure"""
        try:
            repo = git.Repo(self.root_dir)
            files = repo.git.ls_files().split('\n')
            return self._build_tree(files)
        except git.InvalidGitRepositoryError:
            # Not a git repo, fall back to directory listing
            return self._build_tree_from_dir()

    def _build_tree(self, files: List[str]) -> str:
        """Build ASCII tree from file list"""
        tree = {}
        for file in files:
            parts = Path(file).parts
            current = tree
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

        return self._format_tree(tree)

    def _format_tree(self, tree: dict, prefix: str = "") -> str:
        """Format tree as ASCII"""
        lines = []
        items = sorted(tree.items())
        for i, (name, subtree) in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            lines.append(f"{prefix}{current_prefix}{name}")

            if subtree:
                extension_prefix = "    " if is_last else "│   "
                lines.append(self._format_tree(subtree, prefix + extension_prefix))

        return "\n".join(lines)

    def _build_tree_from_dir(self) -> str:
        """Build tree from directory (fallback)"""
        # Simplified implementation
        return str(list(self.root_dir.rglob("*")))
