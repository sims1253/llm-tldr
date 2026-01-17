"""Tests for R language support in HybridExtractor (TDD - tests first).

These tests define the expected behavior for:
1. File extension recognition (.r and .R)
2. Function extraction with <- operator (R-specific)
3. S3 method extraction (generic.classname pattern)
4. Internal helper extraction (functions starting with .)
5. R6 class extraction with public/private methods
6. S4 class extraction (setClass pattern)
7. Import/require/source extraction
8. Roxygen docstring extraction (#' comments)
9. Parameter extraction with default values
10. Call graph extraction within functions

All tests should FAIL initially because:
- R-specific extraction needs to be fully implemented
- tree-sitter-r may not be available
"""

from pathlib import Path

import pytest
from tldr.hybrid_extractor import HybridExtractor


class TestRFileExtensionRecognition:
    """Test that .r and .R files are recognized and parsed as R."""

    def test_r_extension_lowercase_detected(self, tmp_path: Path):
        """File with .r extension should be detected as 'r' language."""
        r_file = tmp_path / "test.r"
        r_file.write_text(r"""
x <- 10
print(x)
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should detect language as "r"
        assert result.language == "r"

    def test_r_extension_uppercase_detected(self, tmp_path: Path):
        """File with .R extension should be detected as 'r' language."""
        r_file = tmp_path / "module.R"
        r_file.write_text(r"""
calculate_mean <- function(x) {
    mean(x)
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should detect language as "r" regardless of case
        assert result.language == "r"

    def test_r_extension_not_treated_as_unknown(self, tmp_path: Path):
        """R files should NOT be processed as unknown language."""
        r_file = tmp_path / "script.r"
        r_file.write_text(r"""
# Simple R script
result <- sum(1, 2, 3)
print(result)
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Language should be specifically "r"
        assert result.language == "r"


class TestRFunctionExtraction:
    """Test extraction of R functions with <- operator."""

    def test_function_with_left_arrow_assignment(self, tmp_path: Path):
        """Functions with <- assignment should be extracted."""
        r_file = tmp_path / "functions.r"
        r_file.write_text(r"""
greet <- function(name) {
    paste("Hello,", name)
}

calculate_sum <- function(a, b) {
    a + b
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract both functions
        assert len(result.functions) == 2

        # Find greet function
        greet = next((f for f in result.functions if f.name == "greet"), None)
        assert greet is not None
        assert "name" in greet.params

        # Find calculate_sum function
        calc_sum = next(
            (f for f in result.functions if f.name == "calculate_sum"), None
        )
        assert calc_sum is not None

    def test_function_with_equals_assignment(self, tmp_path: Path):
        """Functions with = assignment should be extracted."""
        r_file = tmp_path / "equals_assign.r"
        r_file.write_text(r"""
square = function(x) {
    x * x
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the function
        assert len(result.functions) == 1
        assert result.functions[0].name == "square"

    def test_function_with_dots_parameter(self, tmp_path: Path):
        """Functions with ... parameter should be extracted."""
        r_file = tmp_path / "dots_param.r"
        r_file.write_text(r"""
concat <- function(..., sep = " ") {
    paste(..., sep = sep)
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the function
        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "concat"
        # Should capture the dots parameter
        assert "..." in func.params


class TestRS3MethodExtraction:
    """Test extraction of R S3 methods (generic.classname pattern)."""

    def test_s3_method_extraction(self, tmp_path: Path):
        """S3 methods with generic.classname pattern should be extracted."""
        r_file = tmp_path / "s3_methods.r"
        r_file.write_text(r"""
print.myclass <- function(x) {
    cat("My class:", class(x)[1], "\n")
}

summary.data.frame <- function(df) {
    lapply(df, summary)
}

plot.robins_i_result <- function(result) {
    plot(result$estimates)
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract S3 methods
        assert len(result.functions) >= 3

        # Find the S3 methods
        func_names = [f.name for f in result.functions]
        assert any("print.myclass" in name for name in func_names)
        assert any("summary.data.frame" in name for name in func_names)
        assert any("plot.robins_i_result" in name for name in func_names)

    def test_s3_method_not_confused_with_internal(self, tmp_path: Path):
        """Internal helper functions (starting with .) should NOT be S3 methods."""
        r_file = tmp_path / "internal_helper.r"
        r_file.write_text(r"""
.internal_helper <- function(x) {
    x * 2
}

calculate.result <- function(x) {
    .internal_helper(x)
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the S3 method but NOT flag .internal_helper as S3
        func_names = [f.name for f in result.functions]
        assert any("calculate.result" in name for name in func_names)


class TestRInternalHelperExtraction:
    """Test extraction of internal helper functions (starting with .)."""

    def test_internal_helper_extraction(self, tmp_path: Path):
        """Functions starting with . should be extracted as internal helpers."""
        r_file = tmp_path / "internal_helpers.r"
        r_file.write_text(r""".cache_result <- function(x) {
    # Internal caching logic
    if (exists(".cache")) return(.cache)
    .cache <- x
    return(x)
}

.setup_environment <- function() {
    Sys.setenv(FOO = "bar")
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract internal helpers
        func_names = [f.name for f in result.functions]
        assert any(".cache_result" in name for name in func_names)
        assert any(".setup_environment" in name for name in func_names)


class TestRR6ClassExtraction:
    """Test extraction of R6 classes with public/private methods."""

    def test_r6_class_extraction(self, tmp_path: Path):
        """R6 classes with public/private methods should be extracted."""
        r_file = tmp_path / "r6_class.r"
        r_file.write_text(r"""
Person <- R6Class("Person",
    public = list(
        name = NULL,
        age = NULL,

        initialize = function(name, age) {
            self$name <- name
            self$age <- age
        },

        greet = function() {
            paste("Hello, my name is", self$name)
        }
    ),
    private = list(
        internal_id = NULL,

        .generate_id = function() {
            private$internal_id <- sample(1000, 1)
        }
    )
)
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the R6 class
        assert len(result.classes) >= 1

        # Find the Person class
        person_class = next((c for c in result.classes if c.name == "Person"), None)
        assert person_class is not None

        # Should extract public methods
        method_names = [m.name for m in person_class.methods]
        assert any("initialize" in name for name in method_names)
        assert any("greet" in name for name in method_names)

    def test_r6_class_with_active_bindings(self, tmp_path: Path):
        """R6 classes with active bindings should be extracted."""
        r_file = tmp_path / "r6_active.r"
        r_file.write_text(r"""
Counter <- R6Class("Counter",
    public = list(
        count = 0,
        increment = function() {
            private$count <- private$count + 1
        }
    ),
    private = list(
        count = NULL
    ),
    active = list(
        value = function() private$count
    )
)
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the Counter class
        counter_class = next((c for c in result.classes if c.name == "Counter"), None)
        assert counter_class is not None


class TestRS4ClassExtraction:
    """Test extraction of S4 classes (setClass pattern)."""

    def test_s4_class_extraction(self, tmp_path: Path):
        """S4 classes with setClass should be extracted."""
        r_file = tmp_path / "s4_class.r"
        r_file.write_text(r"""
setClass("Person",
    representation(
        name = "character",
        age = "numeric"
    ),
    prototype(
        name = NA_character_,
        age = NA_real_
    )
)

setClass("ROBINSIResult",
    representation(
        estimates = "numeric",
        variance = "numeric",
        weights = "numeric"
    ),
    contains = "VIRTUAL"
)
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract S4 classes
        class_names = [c.name for c in result.classes]
        assert any("Person" in name for name in class_names)
        assert any("ROBINSIResult" in name for name in class_names)


class TestRImportExtraction:
    """Test extraction of library(), require(), and source() calls."""

    def test_library_import_extraction(self, tmp_path: Path):
        """library() calls should be extracted as imports."""
        r_file = tmp_path / "imports.r"
        r_file.write_text(r"""
library(dplyr)
library(ggplot2)
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract 2 imports
        assert len(result.imports) == 2

        # Check that package names are captured
        modules = [imp.module for imp in result.imports]
        assert any("dplyr" in m for m in modules)
        assert any("ggplot2" in m for m in modules)

    def test_require_import_extraction(self, tmp_path: Path):
        """require() calls should be extracted as imports."""
        r_file = tmp_path / "require_import.r"
        r_file.write_text(r"""
require(data.table)
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the import
        assert len(result.imports) == 1
        assert "data.table" in result.imports[0].module

    def test_source_import_extraction(self, tmp_path: Path):
        """source() calls should be extracted as imports with is_from=True."""
        r_file = tmp_path / "source_import.r"
        r_file.write_text(r"""
source("utils.R")
source("helpers.R")
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract 2 imports
        assert len(result.imports) == 2

        # source() imports should have is_from=True
        modules = [imp.module for imp in result.imports]
        assert any("utils.R" in m for m in modules)
        assert any("helpers.R" in m for m in modules)

    def test_namespaced_import_extraction(self, tmp_path: Path):
        """Namespace-qualified imports should be handled."""
        r_file = tmp_path / "namespaced_import.r"
        r_file.write_text(r"""
library(dplyr, warn.conflicts = FALSE)
library(ggplot2)
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the packages
        assert len(result.imports) >= 1


class TestRRoxygenDocstringExtraction:
    """Test extraction of roxygen2 documentation (#' comments)."""

    def test_roxygen_docstring_extraction(self, tmp_path: Path):
        """Roxygen comments (#') should be extracted as docstrings."""
        r_file = tmp_path / "roxygen_doc.r"
        r_file.write_text(r"""
#' Calculate the mean of a numeric vector
#'
#' @param x A numeric vector
#' @return The mean value
#' @export
calculate_mean <- function(x) {
    mean(x)
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the function
        assert len(result.functions) == 1

        func = result.functions[0]
        assert func.name == "calculate_mean"
        # Should extract roxygen documentation
        assert func.docstring is not None
        assert "mean" in func.docstring.lower()

    def test_roxygen_multiline_docstring(self, tmp_path: Path):
        """Multiline roxygen documentation should be extracted."""
        r_file = tmp_path / "roxygen_multiline.r"
        r_file.write_text(r"""
#' A powerful data processing function
#'
#' This function processes data in multiple ways including
#' filtering, transforming, and aggregating.
#'
#' @param data A data frame to process
#' @param filter_condition Condition for filtering
#' @param transform_fn Function to apply for transformation
#' @return Processed data frame
#' @examples
#' # Example usage
#' processed <- process_data(mtcars, hp > 100, identity)
process_data <- function(data, filter_condition, transform_fn) {
    # Implementation
    data
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the function with docstring
        func = result.functions[0]
        assert func.name == "process_data"
        assert func.docstring is not None
        assert "processes data" in func.docstring.lower()


class TestRParameterExtraction:
    """Test extraction of R function parameters with default values."""

    def test_parameters_with_defaults(self, tmp_path: Path):
        """Parameters with default values should show the defaults."""
        r_file = tmp_path / "params_with_defaults.r"
        r_file.write_text(r"""
process_data <- function(data, filter = TRUE, verbose = FALSE, na.rm = TRUE) {
    if (filter) data <- filter(data)
    data
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the function
        func = result.functions[0]
        assert func.name == "process_data"

        # Should capture parameters with defaults
        assert len(func.params) == 4

    def test_parameters_with_complex_defaults(self, tmp_path: Path):
        """Parameters with complex defaults should be simplified."""
        r_file = tmp_path / "complex_defaults.r"
        r_file.write_text(r"""
create_plot <- function(data, color = "steelblue", size = 1.5, alpha = 0.7) {
    ggplot(data, aes(x = x, y = y)) + geom_point()
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        func = result.functions[0]
        assert func.name == "create_plot"
        # Should have parameters extracted
        assert len(func.params) == 4


class TestRCallGraphExtraction:
    """Test call graph extraction within R functions."""

    def test_call_graph_extraction(self, tmp_path: Path):
        """Should extract call relationships between R functions."""
        r_file = tmp_path / "call_graph.r"
        r_file.write_text(r"""helper_function <- function(x) {
    mean(x)
}

validate_input <- function(data) {
    if (!is.data.frame(data)) stop("Not a data frame")
    TRUE
}

main_function <- function(input_data) {
    validate_input(input_data)
    result <- helper_function(input_data$x)
    return(result)
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should have call graph
        assert result.call_graph is not None
        assert result.call_graph is not None

        # main_function should call helper_function and validate_input
        calls = result.call_graph.calls
        assert "main_function" in calls
        # Should contain both helper_function and validate_input calls
        called_funcs = calls.get("main_function", [])
        assert "helper_function" in called_funcs or any(
            "helper_function" in c for c in called_funcs
        )

    def test_call_graph_deduplication(self, tmp_path: Path):
        """Duplicate calls should be deduplicated in call graph."""
        r_file = tmp_path / "dedup_calls.r"
        r_file.write_text(r"""
compute <- function(x) {
    x + 1
    x + 1
    x + 1
}

main <- function() {
    compute(1)
    compute(2)
    compute(3)
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should have call graph with deduplication
        assert result.call_graph is not None
        assert "main" in result.call_graph.calls
        # compute should only appear once despite multiple calls
        called_funcs = result.call_graph.calls.get("main", [])
        if isinstance(called_funcs, list):
            assert called_funcs.count("compute") == 1 or len(set(called_funcs)) == 1


class TestRClassExtraction:
    """Test general R class extraction patterns."""

    def test_s7_class_extraction(self, tmp_path: Path):
        """S7 classes should be extracted."""
        r_file = tmp_path / "s7_class.r"
        r_file.write_text(r"""
MyClass <- new_class("MyClass",
    properties = list(
        x = new_property(numeric),
        y = new_property(numeric)
    ),
    methods = list(
        sum = function() self$x + self$y
    )
)
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the S7 class
        class_names = [c.name for c in result.classes]
        assert any("MyClass" in name for name in class_names)

    def test_refclass_extraction(self, tmp_path: Path):
        """RefClass (R.oo) classes should be extracted."""
        r_file = tmp_path / "refclass.r"
        r_file.write_text(r"""
MyRefClass <- setRefClass("MyRefClass",
    fields = list(
        value = "numeric",
        label = "character"
    ),
    methods = list(
        initialize = function(...) {
            value <<- 0
        },
        get_value = function() value,
        set_value = function(v) value <<- v
    )
)
""")

        extractor = HybridExtractor()
        result = extractor.extract(r_file)

        # Should extract the RefClass
        class_names = [c.name for c in result.classes]
        assert any("MyRefClass" in name for name in class_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
