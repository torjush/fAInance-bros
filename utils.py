"""Utility functions for finance agents."""

import re


def strip_code_blocks(text: str) -> str:
    """Strip markdown code blocks if present."""
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        return match.group(1).strip()
    return text.strip()
