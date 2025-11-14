"""Utilities for loading system prompts from files."""

from pathlib import Path
from typing import Optional


def load_system_prompt(prompt_name: str) -> str:
    """
    Load a system prompt from the prompts directory.

    Args:
        prompt_name: Name of the prompt file (without .txt extension)

    Returns:
        Content of the prompt file

    Raises:
        FileNotFoundError: If prompt file doesn't exist
        UnicodeDecodeError: If file cannot be decoded as UTF-8
        PermissionError: If file cannot be read due to permissions
    """
    prompts_dir = Path(__file__).parent.parent / "prompts"
    prompt_file = prompts_dir / f"{prompt_name}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}\n"
            f"Available prompts should be in: {prompts_dir}"
        )

    try:
        return prompt_file.read_text(encoding='utf-8')
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end,
            f"Failed to decode prompt file {prompt_file} as UTF-8"
        )
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied when reading prompt file: {prompt_file}"
        ) from e


def build_prompt_with_context(
    system_prompt: str,
    user_task: str,
    context: Optional[str] = None
) -> str:
    """
    Build a complete prompt with system instructions, context, and task.

    Args:
        system_prompt: System-level instructions
        user_task: The specific task to perform
        context: Optional codebase or additional context

    Returns:
        Formatted prompt ready for Gemini
    """
    parts = [
        "=== SYSTEM INSTRUCTIONS ===",
        system_prompt,
        ""
    ]

    if context:
        parts.extend([
            "=== CONTEXT ===",
            context,
            ""
        ])

    parts.extend([
        "=== TASK ===",
        user_task
    ])

    return "\n".join(parts)
