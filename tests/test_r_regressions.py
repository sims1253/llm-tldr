"""Regression tests for R-related bug fixes.

These tests verify specific bugs that were fixed don't regress.
"""

import argparse
import pytest
from pathlib import Path

# Skip all tests if tree-sitter-r is not available
pytest.importorskip("tree_sitter_r")


class TestCrossFileCallsNamedArguments:
    """Regression test for named argument parsing in library() calls.

    Bug: library(package = dplyr) would extract "package" instead of "dplyr"
    Fix: Use child_by_field_name("value") to get the actual value
    """

    def test_library_named_argument(self, tmp_path: Path):
        """library(package = dplyr) should extract 'dplyr', not 'package'."""
        from tldr.cross_file_calls import parse_r_imports

        r_file = tmp_path / "test.R"
        r_file.write_text("library(package = dplyr)\n")

        imports = parse_r_imports(r_file)

        assert len(imports) == 1
        assert imports[0]["module"] == "dplyr", (
            f"Expected 'dplyr', got '{imports[0]['module']}'"
        )

    def test_require_named_argument(self, tmp_path: Path):
        """require(package = ggplot2) should extract 'ggplot2'."""
        from tldr.cross_file_calls import parse_r_imports

        r_file = tmp_path / "test.R"
        r_file.write_text("require(package = ggplot2)\n")

        imports = parse_r_imports(r_file)

        assert len(imports) == 1
        assert imports[0]["module"] == "ggplot2"

    def test_library_positional_still_works(self, tmp_path: Path):
        """library(dplyr) should still work (positional argument)."""
        from tldr.cross_file_calls import parse_r_imports

        r_file = tmp_path / "test.R"
        r_file.write_text("library(dplyr)\n")

        imports = parse_r_imports(r_file)

        assert len(imports) == 1
        assert imports[0]["module"] == "dplyr"

    def test_source_named_argument(self, tmp_path: Path):
        """source(file = 'utils.R') should extract 'utils.R'."""
        from tldr.cross_file_calls import parse_r_imports

        r_file = tmp_path / "test.R"
        r_file.write_text("source(file = 'utils.R')\n")

        imports = parse_r_imports(r_file)

        assert len(imports) == 1
        assert imports[0]["module"] == "utils.R"
        assert imports[0]["is_source"]


class TestDiagnosticsCodeInjection:
    """Regression test for code injection in R diagnostics.

    Bug: Path was interpolated into R code, allowing injection
    Fix: Pass path via commandArgs() instead
    """

    def test_path_with_quotes_does_not_break(self, tmp_path: Path):
        """Paths with quotes should not cause code injection."""
        from tldr.diagnostics import get_diagnostics
        import shutil

        # Skip if Rscript not available
        if not shutil.which("Rscript"):
            pytest.skip("Rscript not available")

        # Create a file with a tricky name (quotes in path)
        # Note: On most systems this is valid, but we test the escaping logic
        r_file = tmp_path / "test'file.R"
        r_file.write_text("x <- 1\n")

        # Result may be empty if lintr not installed, but should not crash
        result = get_diagnostics(str(r_file), "r", include_lint=True)
        assert isinstance(result, dict)


class TestCliAutoSentinel:
    """Regression test for --lang auto sentinel.

    Bug: --lang python was treated as auto-detect, couldn't force Python
    Fix: Use "auto" as default sentinel
    """

    def test_lang_default_is_auto(self):
        """The default for --lang should be 'auto', not 'python'."""
        # Import the CLI module and access the real parser
        from tldr.cli import _build_parser

        # Build the parser and get the context subparser
        parser = _build_parser()

        # Find the context subparser by looking at subparsers
        subparsers_action = None
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                subparsers_action = action
                break

        assert subparsers_action is not None, "No subparsers found in parser"

        # Get the context subparser
        ctx_p = subparsers_action.choices.get("context")
        assert ctx_p is not None, "context subparser not found"

        # Find the --lang argument in the context subparser
        lang_action = None
        for action in ctx_p._actions:
            if hasattr(action, "dest") and action.dest == "lang":
                lang_action = action
                break

        assert lang_action is not None, "--lang argument not found in context subparser"
        assert lang_action.default == "auto", (
            f"--lang should default to 'auto', got '{lang_action.default}'"
        )


class TestSemanticRExtractors:
    """Regression test for R in semantic extractor maps.

    Bug: R extractors weren't in semantic.py extractor maps
    Fix: Added 'r' to CFG and DFG extractor maps
    """

    def test_r_cfg_summary_works(self, tmp_path: Path):
        """R functions should get CFG summaries in semantic embeddings."""
        from tldr.semantic import _get_cfg_summary

        r_file = tmp_path / "test.R"
        r_file.write_text("""
test_func <- function(x) {
    if (x > 0) {
        return(x)
    } else {
        return(-x)
    }
}
""")

        summary = _get_cfg_summary(r_file, "test_func", "r")

        # Should return a non-empty summary with complexity info
        assert summary != "", "R function should have CFG summary"
        assert "complexity" in summary.lower() or "blocks" in summary.lower()

    def test_r_dfg_summary_works(self, tmp_path: Path):
        """R functions should get DFG summaries in semantic embeddings."""
        from tldr.semantic import _get_dfg_summary

        r_file = tmp_path / "test.R"
        r_file.write_text("""
test_func <- function(x) {
    y <- x * 2
    z <- y + 1
    return(z)
}
""")

        summary = _get_dfg_summary(r_file, "test_func", "r")

        # Should return a non-empty summary with variable info
        assert summary != "", "R function should have DFG summary"
        assert "vars" in summary.lower() or "def-use" in summary.lower()
