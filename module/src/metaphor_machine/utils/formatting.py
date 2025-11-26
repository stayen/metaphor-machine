"""
Formatting utilities for different AI music platforms.

Each platform has specific requirements and character limits.
This module handles platform-specific formatting.
"""

from __future__ import annotations

from metaphor_machine.core.metaphor import Metaphor, MetaphorChain


# Platform-specific character limits
SUNO_STYLE_LIMIT = 120
SUNO_LYRICS_LIMIT = 3000
PRODUCER_AI_SOUND_LIMIT = 500  # Approximate


def truncate_to_limit(text: str, limit: int, ellipsis: str = "...") -> str:
    """
    Truncate text to fit within a character limit.

    Attempts to truncate at word boundaries when possible.

    Args:
        text: Text to truncate
        limit: Maximum character count
        ellipsis: String to append when truncating (default: "...")

    Returns:
        Truncated text, or original if within limit

    Example:
        >>> truncate_to_limit("hello world example", 15)
        'hello world...'
    """
    if len(text) <= limit:
        return text

    # Account for ellipsis length
    target_len = limit - len(ellipsis)
    if target_len <= 0:
        return ellipsis[:limit]

    # Try to break at word boundary
    truncated = text[:target_len]
    last_space = truncated.rfind(" ")

    if last_space > target_len * 0.5:  # Only if we keep > 50%
        truncated = truncated[:last_space]

    return truncated.rstrip() + ellipsis


def format_for_suno(
    item: Metaphor | MetaphorChain,
    enforce_limit: bool = True,
) -> str:
    """
    Format a metaphor or chain for Suno's Style of Music field.

    Suno has a 120-character limit on style prompts.

    Args:
        item: Metaphor or MetaphorChain to format
        enforce_limit: If True, truncate to fit 120 chars

    Returns:
        Formatted string for Suno

    Example:
        >>> format_for_suno(metaphor)
        'lo-fi boom-bap, breathy confessions, pulsing 808s, subway-tunnel echo...'
    """
    if isinstance(item, MetaphorChain):
        # For chains, we need to be more aggressive about truncation
        # Only include the most essential elements
        text = item.to_suno_style(separator=" → ")
    else:
        text = str(item)

    if enforce_limit:
        return truncate_to_limit(text, SUNO_STYLE_LIMIT)
    return text


def format_for_producer_ai(
    item: Metaphor | MetaphorChain,
    enforce_limit: bool = True,
) -> str:
    """
    Format a metaphor or chain for Producer.ai's Sound Prompt field.

    Producer.ai allows longer prompts than Suno.

    Args:
        item: Metaphor or MetaphorChain to format
        enforce_limit: If True, truncate to platform limit

    Returns:
        Formatted string for Producer.ai

    Example:
        >>> format_for_producer_ai(chain)
        'Intro: cinematic pop-ballad... Mid: voice opens... Outro: melody dissolves...'
    """
    if isinstance(item, MetaphorChain):
        text = item.to_suno_style(separator=" | ")
    else:
        text = str(item)

    if enforce_limit:
        return truncate_to_limit(text, PRODUCER_AI_SOUND_LIMIT)
    return text


def format_compact(metaphor: Metaphor) -> str:
    """
    Format a metaphor in compact form, prioritizing most impactful slots.

    Useful when character limits are very tight.

    Args:
        metaphor: Metaphor to format

    Returns:
        Compact string with only genre, gesture, and anchor
    """
    return f"{metaphor.genre_anchor}, {metaphor.intimate_gesture}, {metaphor.emotional_anchor}"


def format_expanded(metaphor: Metaphor) -> str:
    """
    Format a metaphor with slot labels for clarity.

    Useful for debugging and documentation.

    Args:
        metaphor: Metaphor to format

    Returns:
        Labeled multi-line string
    """
    return (
        f"Genre: {metaphor.genre_anchor}\n"
        f"Gesture: {metaphor.intimate_gesture}\n"
        f"Tension: {metaphor.dynamic_tension}\n"
        f"Bridge: {metaphor.sensory_bridge}\n"
        f"Anchor: {metaphor.emotional_anchor}"
    )


def format_chain_multiline(chain: MetaphorChain) -> str:
    """
    Format a chain with each section on its own line.

    Args:
        chain: MetaphorChain to format

    Returns:
        Multi-line string with labeled sections
    """
    return (
        f"[INTRO]\n{chain.intro}\n\n"
        f"[MID]\n{chain.mid}\n\n"
        f"[OUTRO]\n{chain.outro}"
    )
