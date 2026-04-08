"""Token counting utilities for the Tok runtime."""

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True

    def count_tokens(text: str) -> int:
        """Count tokens using cl100k_base encoding with fallback."""
        if not text:
            return 0
        return len(_ENC.encode(text, disallowed_special=()))

except ImportError:
    _HAS_TIKTOKEN = False

    def count_tokens(text: str) -> int:
        """Count tokens using fallback character-based approximation."""
        if not text:
            return 0
        return len(text) // 4


__all__ = ["_HAS_TIKTOKEN", "count_tokens"]
