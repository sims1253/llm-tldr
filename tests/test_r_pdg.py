"""Tests for R Program Dependency Graph extraction (TDD - write tests first).

R PDG combines:
- Control Flow Graph (CFG) - branches, loops, continue/break
- Data Flow Graph (DFG) - variable definitions and uses

These tests define expected behavior for extract_r_pdg().

R-specific features tested:
- <- for assignment (not =)
- { } for blocks
- next statement (R's continue equivalent)
- repeat loop (executes at least once, R-specific)
- switch statement (R-specific, different from C-style)
- for (i in 1:n) loop syntax
- if-else if chains
"""

import pytest


class TestRPDGBasic:
    """Tests for basic R PDG extraction with <- assignment."""

    def test_r_pdg_simple_function(self):
        """Should extract PDG for simple R function with <- assignment."""
        from tldr.pdg_extractor import extract_r_pdg

        code = """simple <- function(x) {
    y <- x * 2
    return(y)
}
"""
        pdg = extract_r_pdg(code, "simple")

        assert pdg is not None
        # PDG should have data edges (control edges for branches)
        data_edges = [e for e in pdg.edges if e.dep_type == "data"]
        assert len(data_edges) > 0
        # Data edge from x parameter to y definition via <-
        data_vars = {edge.label for edge in data_edges}
        assert "x" in data_vars or "y" in data_vars

    def test_r_pdg_with_next(self):
        """Should handle next statement (R's continue equivalent) in PDG."""
        from tldr.pdg_extractor import extract_r_pdg

        code = """filter_odd <- function(n) {
    total <- 0
    for (i in 1:n) {
        if (i %% 2 == 0) {
            next
        }
        total <- total + i
    }
    return(total)
}
"""
        pdg = extract_r_pdg(code, "filter_odd")

        assert pdg is not None
        # Next should create control flow edges (continue edge)
        control_edges = [e for e in pdg.edges if e.dep_type == "control"]
        assert len(control_edges) >= 3  # At least entry, loop, next
        # Data edges for total at DFG level
        total_edges = [e for e in pdg.dfg.dataflow_edges if e.var_name == "total"]
        assert len(total_edges) > 0

    def test_r_pdg_with_break(self):
        """Should handle break statement in R PDG."""
        from tldr.pdg_extractor import extract_r_pdg

        code = """find_first_negative <- function(values) {
    for (v in values) {
        if (v < 0) {
            break
        }
    }
    return(v)
}
"""
        pdg = extract_r_pdg(code, "find_first_negative")

        assert pdg is not None
        # Break should create control dependency edge
        break_edges = [e for e in pdg.edges if e.dep_type == "control" and "break" in e.label]
        assert len(break_edges) >= 1


class TestRPDGControlFlow:
    """Tests for R control flow statements (if-else, loops)."""

    def test_r_pdg_if_else_control_dependencies(self):
        """Should extract control dependencies for if-else statements."""
        from tldr.pdg_extractor import extract_r_pdg

        code = """absolute_value <- function(x) {
    if (x > 0) {
        return(x)
    } else {
        return(-x)
    }
}
"""
        pdg = extract_r_pdg(code, "absolute_value")

        assert pdg is not None
        # If-else creates branch control edges
        control_edges = [e for e in pdg.edges if e.dep_type == "control"]
        assert len(control_edges) >= 3  # entry, true branch, false branch

        # Data dependencies for x
        x_edges = [e for e in pdg.edges if e.dep_type == "data" and e.label == "x"]
        assert len(x_edges) > 0

    def test_r_pdg_for_loop_iterator(self):
        """Should track iterator variable in for loop."""
        from tldr.pdg_extractor import extract_r_pdg

        code = """sum_1_to_n <- function(n) {
    total <- 0
    for (i in 1:n) {
        total <- total + i
    }
    return(total)
}
"""
        pdg = extract_r_pdg(code, "sum_1_to_n")

        assert pdg is not None
        # For loop creates loop control edge
        control_edges = [e for e in pdg.edges if e.dep_type == "control"]
        assert len(control_edges) >= 2  # entry, loop back edge

        # Iterator variable i should have data flow
        i_edges = [e for e in pdg.dfg.dataflow_edges if e.var_name == "i"]
        assert len(i_edges) > 0

    def test_r_pdg_while_loop(self):
        """Should handle while loop control dependencies."""
        from tldr.pdg_extractor import extract_r_pdg

        code = """countdown <- function(n) {
    while (n > 0) {
        n <- n - 1
    }
    return(n)
}
"""
        pdg = extract_r_pdg(code, "countdown")

        assert pdg is not None
        # While loop has condition check and back edge
        control_edges = [e for e in pdg.edges if e.dep_type == "control"]
        assert len(control_edges) >= 2  # condition check, back edge

        # Data flow for n
        n_edges = [e for e in pdg.edges if e.dep_type == "data" and e.label == "n"]
        assert len(n_edges) > 0

    def test_r_pdg_repeat_loop(self):
        """Should handle repeat loop (R-specific, executes at least once).

        Repeat loop is R-specific - it executes the body before checking
        the exit condition via break. Unlike while/for, the condition is
        not in the loop header.
        """
        from tldr.pdg_extractor import extract_r_pdg

        code = """repeat_example <- function(limit) {
    x <- 0
    repeat {
        x <- x + 1
        if (x >= limit) {
            break
        }
    }
    return(x)
}
"""
        pdg = extract_r_pdg(code, "repeat_example")

        assert pdg is not None
        # Repeat loop with break
        break_edges = [e for e in pdg.edges if e.dep_type == "control" and "break" in e.label]
        assert len(break_edges) >= 1

        # Control flow for repeat loop structure
        control_edges = [e for e in pdg.edges if e.dep_type == "control"]
        assert len(control_edges) >= 2  # body flow, break exit

        # Data flow for x
        x_edges = [e for e in pdg.dfg.dataflow_edges if e.var_name == "x"]
        assert len(x_edges) >= 2  # x <- 0, x <- x + 1

    def test_r_pdg_switch_statement(self):
        """Should handle switch statement (R-specific).

        R's switch() is different from C-style switch - it selects from
        a list of alternatives by position or name, not by constant values.
        """
        from tldr.pdg_extractor import extract_r_pdg

        code = """switch_example <- function(type) {
    result <- switch(type,
        "a" = { 1 },
        "b" = { 2 },
        "c" = { 3 },
        { 0 }  # default
    )
    return(result)
}
"""
        pdg = extract_r_pdg(code, "switch_example")

        assert pdg is not None
        # Switch creates multiple branch control edges
        control_edges = [e for e in pdg.edges if e.dep_type == "control"]
        assert len(control_edges) >= 4  # at least 4 case branches

        # Data flow for result variable
        result_edges = [e for e in pdg.dfg.dataflow_edges if e.var_name == "result"]
        assert len(result_edges) > 0


class TestRPDGNestedControlFlow:
    """Tests for nested R control flow constructs."""

    def test_r_pdg_nested_control_flow(self):
        """Should handle nested if inside for loop."""
        from tldr.pdg_extractor import extract_r_pdg

        code = """count_positive <- function(matrix) {
    count <- 0
    for (i in 1:nrow(matrix)) {
        for (j in 1:ncol(matrix)) {
            if (matrix[i, j] > 0) {
                count <- count + 1
            }
        }
    }
    return(count)
}
"""
        pdg = extract_r_pdg(code, "count_positive")

        assert pdg is not None
        # Nested loops add complexity
        control_edges = [e for e in pdg.edges if e.dep_type == "control"]
        assert len(control_edges) >= 4  # outer loop, inner loop, if branch

        # Multiple data variables
        count_edges = [e for e in pdg.dfg.dataflow_edges if e.var_name == "count"]
        assert len(count_edges) > 0

        i_edges = [e for e in pdg.dfg.dataflow_edges if e.var_name == "i"]
        assert len(i_edges) > 0


class TestRPDGErrorHandling:
    """Tests for error handling in R PDG extraction."""

    def test_r_pdg_function_not_found(self):
        """Should return None for non-existent function."""
        from tldr.pdg_extractor import extract_r_pdg

        code = """exists <- function() {
    print("exists")
}
"""
        pdg = extract_r_pdg(code, "nonexistent")

        assert pdg is None


class TestRPDGDataDependencies:
    """Tests for R-specific data dependency patterns."""

    def test_r_pdg_assignment_chain(self):
        """Should track data flow through <- assignments."""
        from tldr.pdg_extractor import extract_r_pdg

        code = """accumulate <- function(n) {
    x <- 0
    x <- x + 1
    x <- x + 2
    return(x)
}
"""
        pdg = extract_r_pdg(code, "accumulate")

        assert pdg is not None
        # Multiple data edges for x showing the chain
        x_edges = [e for e in pdg.dfg.dataflow_edges if e.var_name == "x"]
        # Should have edges for: init, first update, second update
        assert len(x_edges) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
