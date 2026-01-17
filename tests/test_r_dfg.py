"""Tests for R Data Flow Graph extraction (TDD - tests written before implementation).

These tests define the expected behavior for Data Flow Graph extraction
from R code, including R-specific features:
- Leftward assignment with <- (most common in R)
- Rightward assignment with -> (R-specific)
- Global assignment with <<- (R-specific)
- Default parameter values in functions
- For loop variable tracking
- NULL/NA handling in assignments

The function `extract_r_dfg` exists but these tests verify R-specific behavior.
"""

import pytest


# =============================================================================
# Test 1: Basic Definition and Use with <- Assignment
# =============================================================================


def test_r_dfg_basic_def_use_leftward():
    """Basic variable definition and use with <- assignment should be tracked.

    R uses <- as the primary assignment operator (not = or ==).
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
example <- function() {
    x <- 10
    y <- x + 5
    return(y)
}
"""
    dfg = extract_r_dfg(code, "example")

    assert dfg is not None
    assert dfg.function_name == "example"

    # Find x definition
    x_defs = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "definition"]
    assert len(x_defs) >= 1, "x should have at least one definition"

    # Find x use
    x_uses = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "use"]
    assert len(x_uses) >= 1, "x should have at least one use"

    # Find y definition and use
    y_defs = [r for r in dfg.var_refs if r.name == "y" and r.ref_type == "definition"]
    y_uses = [r for r in dfg.var_refs if r.name == "y" and r.ref_type == "use"]
    assert len(y_defs) >= 1, "y should have at least one definition"
    assert len(y_uses) >= 1, "y should have at least one use (in return)"


# =============================================================================
# Test 2: Rightward Assignment with -> Operator (R-specific)
# =============================================================================


def test_r_dfg_rightward_assignment():
    """Rightward assignment with -> operator should be tracked correctly.

    R supports rightward assignment: 10 -> x
    This is R-specific and differs from most other languages.
    The variable is on the right side of the operator.
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
rightward_example <- function() {
    result <- 42
    100 -> x
    y <- x * 2
    return(y)
}
"""
    dfg = extract_r_dfg(code, "rightward_example")

    assert dfg is not None

    # x is defined via rightward assignment
    x_defs = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "definition"]
    assert len(x_defs) >= 1, "x should be defined via rightward assignment"

    # x is used in the expression
    x_uses = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "use"]
    assert len(x_uses) >= 1, "x should be used in y <- x * 2"


# =============================================================================
# Test 3: Global Assignment with <<- (R-specific)
# =============================================================================


def test_r_dfg_global_assignment():
    """Global assignment with <<- should be tracked.

    R's <<- operator assigns to a variable in the parent environment,
    commonly used for creating "global" variables within functions.
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
global_example <- function() {
    counter <<- 0
    increment <- function() {
        counter <<- counter + 1
        return(counter)
    }
    return(increment)
}
"""
    dfg = extract_r_dfg(code, "global_example")

    assert dfg is not None

    # counter should be defined with <<-
    counter_defs = [
        r for r in dfg.var_refs if r.name == "counter" and r.ref_type == "definition"
    ]
    assert len(counter_defs) >= 1, "counter should be defined with <<-"

    # counter is used (read) before being updated
    counter_uses = [
        r for r in dfg.var_refs if r.name == "counter" and r.ref_type == "use"
    ]
    assert len(counter_uses) >= 1, "counter should be used when incrementing"


# =============================================================================
# Test 4: Function Parameters with Default Values
# =============================================================================


def test_r_dfg_parameters_with_defaults():
    """Function parameters with default values should be tracked as definitions.

    R function parameters can have default values like: function(x = 10)
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
param_example <- function(x = 5, y = 10) {
    result <- x + y
    return(result)
}
"""
    dfg = extract_r_dfg(code, "param_example")

    assert dfg is not None

    # x is defined as parameter
    x_defs = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "definition"]
    assert len(x_defs) >= 1, "x should be defined as parameter"

    # y is defined as parameter
    y_defs = [r for r in dfg.var_refs if r.name == "y" and r.ref_type == "definition"]
    assert len(y_defs) >= 1, "y should be defined as parameter"

    # Both x and y are used in the expression
    x_uses = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "use"]
    y_uses = [r for r in dfg.var_refs if r.name == "y" and r.ref_type == "use"]
    assert len(x_uses) >= 1, "x should be used in x + y"
    assert len(y_uses) >= 1, "y should be used in x + y"


# =============================================================================
# Test 5: Variable Tracking Across If Statement
# =============================================================================


def test_r_dfg_variable_across_if():
    """Variable tracking should work across if-else statements.

    Definitions in both branches should be tracked.
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
if_example <- function(condition) {
    if (condition) {
        x <- 10
    } else {
        x <- 20
    }
    return(x)
}
"""
    dfg = extract_r_dfg(code, "if_example")

    assert dfg is not None

    # x should have definitions in both branches
    x_defs = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "definition"]
    assert len(x_defs) >= 2, (
        f"x should have at least 2 definitions (if/else), got {len(x_defs)}"
    )

    # x is used in return
    x_uses = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "use"]
    assert len(x_uses) >= 1, "x should be used in return"


# =============================================================================
# Test 6: For Loop Variable Tracking
# =============================================================================


def test_r_dfg_for_loop_variable():
    """For loop variable should be tracked as a definition.

    R for loops: for (i in 1:10) { ... }
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
for_example <- function() {
    total <- 0
    for (i in 1:10) {
        total <- total + i
    }
    return(total)
}
"""
    dfg = extract_r_dfg(code, "for_example")

    assert dfg is not None

    # i is defined by for loop
    i_defs = [r for r in dfg.var_refs if r.name == "i" and r.ref_type == "definition"]
    assert len(i_defs) >= 1, "i should be defined by for loop"

    # i is used in the loop body
    i_uses = [r for r in dfg.var_refs if r.name == "i" and r.ref_type == "use"]
    assert len(i_uses) >= 1, "i should be used in total + i"

    # total is updated in loop
    total_updates = [
        r for r in dfg.var_refs if r.name == "total" and r.ref_type == "definition"
    ]
    assert len(total_updates) >= 2, "total should have initial def + loop updates"


# =============================================================================
# Test 7: Compound Assignments (Multiple Variables)
# =============================================================================


def test_r_dfg_compound_assignment():
    """Multiple variable assignments should all be tracked.

    R supports assigning multiple variables at once or in sequence.
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
compound_example <- function(a, b) {
    x <- a * 2
    y <- b * 3
    z <- x + y
    return(z)
}
"""
    dfg = extract_r_dfg(code, "compound_example")

    assert dfg is not None

    # All variables should be tracked
    names = {r.name for r in dfg.var_refs}
    assert "x" in names, "x should be tracked"
    assert "y" in names, "y should be tracked"
    assert "z" in names, "z should be tracked"

    # x should have definition and use
    x_defs = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "definition"]
    x_uses = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "use"]
    assert len(x_defs) >= 1, "x should be defined"
    assert len(x_uses) >= 1, "x should be used in z <- x + y"

    # y should have definition and use
    y_defs = [r for r in dfg.var_refs if r.name == "y" and r.ref_type == "definition"]
    y_uses = [r for r in dfg.var_refs if r.name == "y" and r.ref_type == "use"]
    assert len(y_defs) >= 1, "y should be defined"
    assert len(y_uses) >= 1, "y should be used in z <- x + y"


# =============================================================================
# Test 8: Multiple Parameters in Function
# =============================================================================


def test_r_dfg_multiple_parameters():
    """Multiple function parameters should all be tracked as definitions.

    R functions can have any number of parameters.
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
multi_param <- function(a, b, c, d) {
    result <- a + b + c + d
    return(result)
}
"""
    dfg = extract_r_dfg(code, "multi_param")

    assert dfg is not None

    # All parameters should be definitions
    for param_name in ["a", "b", "c", "d"]:
        defs = [
            r
            for r in dfg.var_refs
            if r.name == param_name and r.ref_type == "definition"
        ]
        assert len(defs) >= 1, f"{param_name} should be defined as parameter"


# =============================================================================
# Test 9: Variable Scope in Nested Blocks
# =============================================================================


def test_r_dfg_nested_scope():
    """Variables should be tracked correctly in nested blocks.

    R supports nested {} blocks within functions.
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
nested_example <- function(x) {
    result <- x * 2
    {
        inner_result <- result + 1
        final <- inner_result * 3
    }
    return(result)
}
"""
    dfg = extract_r_dfg(code, "nested_example")

    assert dfg is not None

    # result is defined in outer block and used in return
    result_defs = [
        r for r in dfg.var_refs if r.name == "result" and r.ref_type == "definition"
    ]
    result_uses = [
        r for r in dfg.var_refs if r.name == "result" and r.ref_type == "use"
    ]
    assert len(result_defs) >= 1, "result should be defined"
    assert len(result_uses) >= 1, "result should be used in nested block or return"

    # inner_result and final should be tracked in inner block
    names = {r.name for r in dfg.var_refs}
    assert "inner_result" in names, "inner_result should be tracked"
    assert "final" in names, "final should be tracked"


# =============================================================================
# Test 10: NULL and NA Handling in Assignments
# =============================================================================


def test_r_dfg_null_na_handling():
    """NULL and NA values in assignments should be handled correctly.

    R uses NULL for empty/missing objects and NA for missing values.
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
na_example <- function() {
    x <- NA
    y <- NULL
    z <- c(1, 2, NA, 4)
    has_na <- any(is.na(z))
    return(has_na)
}
"""
    dfg = extract_r_dfg(code, "na_example")

    assert dfg is not None

    # Variables assigned NA/NULL should still be tracked
    x_defs = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "definition"]
    assert len(x_defs) >= 1, "x assigned NA should be tracked"

    y_defs = [r for r in dfg.var_refs if r.name == "y" and r.ref_type == "definition"]
    assert len(y_defs) >= 1, "y assigned NULL should be tracked"

    # z is defined and used
    z_defs = [r for r in dfg.var_refs if r.name == "z" and r.ref_type == "definition"]
    z_uses = [r for r in dfg.var_refs if r.name == "z" and r.ref_type == "use"]
    assert len(z_defs) >= 1, "z should be defined"
    assert len(z_uses) >= 1, "z should be used in is.na(z)"


# =============================================================================
# Test 11: Rightward Global Assignment ->>
# =============================================================================


def test_r_dfg_rightward_global_assignment():
    """Rightward global assignment ->> should be tracked.

    R supports: value ->> variable (rightward global assignment).
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
global_rightward <- function() {
    result <- calculate_value()
    result ->> global_var
    return(global_var)
}
"""
    dfg = extract_r_dfg(code, "global_rightward")

    assert dfg is not None

    # global_var should be defined via ->>
    global_defs = [
        r for r in dfg.var_refs if r.name == "global_var" and r.ref_type == "definition"
    ]
    assert len(global_defs) >= 1, "global_var should be defined via ->>"


# =============================================================================
# Test 12: Function Not Found
# =============================================================================


def test_r_dfg_function_not_found():
    """Should return empty DFG when function not found (not raise)."""
    from tldr.dfg_extractor import extract_r_dfg

    code = """
existing_func <- function() {
    x <- 10
    return(x)
}
"""
    dfg = extract_r_dfg(code, "nonexistent")

    # Following pattern from other languages: return empty DFG, not raise
    assert dfg is not None
    assert dfg.function_name == "nonexistent"
    assert len(dfg.var_refs) == 0


# =============================================================================
# Test 13: Assignment with = Operator
# =============================================================================


def test_r_dfg_equals_assignment():
    """R's = operator should also work for assignment.

    While <- is preferred, R also accepts = for assignment.
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
equals_example <- function() {
    x = 10
    y = x * 2
    return(y)
}
"""
    dfg = extract_r_dfg(code, "equals_example")

    assert dfg is not None

    # x should be defined with =
    x_defs = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "definition"]
    assert len(x_defs) >= 1, "x should be defined with ="

    # x is used
    x_uses = [r for r in dfg.var_refs if r.name == "x" and r.ref_type == "use"]
    assert len(x_uses) >= 1, "x should be used in y = x * 2"


# =============================================================================
# Test 14: While Loop Variable Tracking
# =============================================================================


def test_r_dfg_while_loop():
    """Variables in while loops should be tracked correctly."""
    from tldr.dfg_extractor import extract_r_dfg

    code = """
while_example <- function(n) {
    count <- 0
    while (count < n) {
        count <- count + 1
    }
    return(count)
}
"""
    dfg = extract_r_dfg(code, "while_example")

    assert dfg is not None

    # count is defined initially and updated in loop
    count_defs = [
        r for r in dfg.var_refs if r.name == "count" and r.ref_type == "definition"
    ]
    assert len(count_defs) >= 2, "count should have initial def + loop updates"

    # count is used in condition and increment
    count_uses = [r for r in dfg.var_refs if r.name == "count" and r.ref_type == "use"]
    assert len(count_uses) >= 1, "count should be used in condition"


# =============================================================================
# Test 15: Dataflow Edges for R
# =============================================================================


def test_r_dfg_dataflow_edges():
    """Dataflow edges should connect definitions to uses."""
    from tldr.dfg_extractor import extract_r_dfg

    code = """
flow_example <- function() {
    a <- 10
    b <- a + 5
    c <- b * 2
    return(c)
}
"""
    dfg = extract_r_dfg(code, "flow_example")

    assert dfg is not None
    assert len(dfg.dataflow_edges) >= 1, "Should have dataflow edges"

    # Check that edges connect the right variables
    edge_vars = {edge.var_name for edge in dfg.dataflow_edges}
    assert "a" in edge_vars or "b" in edge_vars, "Edges should track variable flow"


# =============================================================================
# Test 16: Repeat Loop Variable Tracking
# =============================================================================


def test_r_dfg_repeat_loop():
    """Variables in repeat loops should be tracked correctly.

    R's repeat loop requires explicit break condition.
    """
    from tldr.dfg_extractor import extract_r_dfg

    code = """
repeat_example <- function(limit) {
    i <- 1
    repeat {
        if (i >= limit) break
        i <- i + 1
    }
    return(i)
}
"""
    dfg = extract_r_dfg(code, "repeat_example")

    assert dfg is not None

    # i is defined and updated in loop
    i_defs = [r for r in dfg.var_refs if r.name == "i" and r.ref_type == "definition"]
    assert len(i_defs) >= 2, "i should have initial def + loop updates"

    # i is used in condition
    i_uses = [r for r in dfg.var_refs if r.name == "i" and r.ref_type == "use"]
    assert len(i_uses) >= 1, "i should be used in condition"
