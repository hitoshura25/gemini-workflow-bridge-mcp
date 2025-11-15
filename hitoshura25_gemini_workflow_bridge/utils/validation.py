"""Utilities for input validation."""

from typing import Any, List


def validate_enum_parameter(
    param_value: Any,
    param_name: str,
    valid_values: List[str]
) -> tuple[bool, str]:
    """
    Validate that a parameter value is within allowed enum values.

    Args:
        param_value: The value to validate
        param_name: Name of the parameter (for error messages)
        valid_values: List of allowed values

    Returns:
        Tuple of (is_valid, error_message)
        If valid: (True, "")
        If invalid: (False, "error message")
    """
    if param_value not in valid_values:
        error_msg = f"Invalid {param_name}. Must be one of: {valid_values}"
        return False, error_msg
    return True, ""
