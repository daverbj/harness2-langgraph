"""A deliberately incomplete implementation for the coding-agent walkthrough."""


def clamp(value, minimum, maximum):
    """Keep value between inclusive bounds; reject reversed bounds."""
    return max(minimum, value)
