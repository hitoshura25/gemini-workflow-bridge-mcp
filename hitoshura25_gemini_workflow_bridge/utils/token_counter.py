"""Token counting utilities for measuring compression ratios."""

import re
from typing import Union


def count_tokens(text: str) -> int:
    """
    Estimate token count for a given text.

    This is a simple approximation based on word count and punctuation.
    For more accurate counting, consider using a proper tokenizer.

    Args:
        text: Text to count tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Simple token estimation: words + punctuation marks
    words = len(text.split())

    # Count punctuation and special characters
    special_chars = len(re.findall(r'[^\w\s]', text))

    # Rough estimate: 1 word ≈ 1.3 tokens on average
    return int(words * 1.3 + special_chars * 0.5)


def estimate_compression_ratio(
    input_text: Union[str, int],
    output_text: Union[str, int]
) -> float:
    """
    Calculate compression ratio between input and output.

    Args:
        input_text: Input text or token count
        output_text: Output text or token count

    Returns:
        Compression ratio (e.g., 174.0 means 174:1 compression)
        Returns 0.0 if either input or output is 0
    """
    input_tokens = count_tokens(input_text) if isinstance(input_text, str) else input_text
    output_tokens = count_tokens(output_text) if isinstance(output_text, str) else output_text

    # Handle edge cases
    if input_tokens == 0 or output_tokens == 0:
        return 0.0

    return round(input_tokens / output_tokens, 1)


def format_token_stats(
    input_tokens: int,
    output_tokens: int,
    analysis_time: float
) -> dict:
    """
    Format token statistics for metadata.

    Args:
        input_tokens: Number of input tokens analyzed
        output_tokens: Number of output tokens generated
        analysis_time: Analysis time in seconds

    Returns:
        Dictionary with formatted statistics
    """
    compression_ratio = estimate_compression_ratio(input_tokens, output_tokens)

    return {
        "tokens_analyzed": input_tokens,
        "tokens_returned": output_tokens,
        "compression_ratio": compression_ratio,
        "analysis_time_seconds": round(analysis_time, 2)
    }
