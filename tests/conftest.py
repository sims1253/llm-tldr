"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Create tree_sitter_r module wrapper if using tree_sitter_language_pack
# This allows the codebase to work with either the old tree_sitter_r package
# or the newer tree_sitter_language_pack

try:
    import tree_sitter_r
except ImportError:
    # Try to create a wrapper using tree_sitter_language_pack
    try:
        from tree_sitter_language_pack import get_binding

        class _TreeSitterRWrapper:
            """Wrapper that provides tree_sitter_r.language() interface."""

            @staticmethod
            def language():
                """Return the R language binding."""
                return get_binding("r")

        # Create and register the wrapper module
        import types

        tree_sitter_r = types.ModuleType("tree_sitter_r")
        tree_sitter_r.language = _TreeSitterRWrapper.language

        sys.modules["tree_sitter_r"] = tree_sitter_r
    except ImportError:
        # tree_sitter_language_pack not available either
        pass
