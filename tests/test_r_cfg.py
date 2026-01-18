"""Tests for R CFG extraction (TDD - tests written before implementation).

These tests define the expected behavior for Control Flow Graph extraction
from R code, including R-specific features:
- <- for assignment
- { } for blocks
- switch statement (R-specific)
- next statement (R's continue equivalent)
- repeat loop (executes at least once)
- for (i in 1:10) loop syntax
- if-else if chain (R's elif equivalent)

The function `extract_r_cfg` does NOT exist yet - these tests must FAIL.
"""

import pytest


# =============================================================================
# Test 1: Simple Function CFG
# =============================================================================


def test_r_cfg_simple_function():
    """Simple R function with linear code should produce linear CFG with entry/exit."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
test_func <- function(x) {
    y <- x * 2
    return(y)
}
"""
    cfg = extract_r_cfg(code, "test_func")

    # Should have entry and exit blocks
    assert cfg is not None
    assert len(cfg.blocks) >= 1  # At least one block (entry/return)

    # Cyclomatic complexity for linear code is 1
    assert cfg.cyclomatic_complexity == 1

    # Function name should be extracted
    assert cfg.function_name == "test_func"

    # Should have at least one exit block
    assert len(cfg.exit_block_ids) >= 1


# =============================================================================
# Test 2: If-Else Statement
# =============================================================================


def test_r_cfg_if_else():
    """If-else should create branch in CFG."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
abs_func <- function(x) {
    if (x > 0) {
        return(x)
    } else {
        return(-x)
    }
}
"""
    cfg = extract_r_cfg(code, "abs_func")

    assert cfg is not None

    # If with else: complexity = 2 (one decision point)
    assert cfg.cyclomatic_complexity == 2

    # Should have: entry -> condition -> then/else -> exit
    # At least 3 edges: entry->cond, cond->then, cond->else
    assert len(cfg.edges) >= 3


# =============================================================================
# Test 3: Else If Chain (R's elif equivalent)
# =============================================================================


def test_r_cfg_else_if_chain():
    """Else if chain should create multiple decision points in CFG."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
grade <- function(score) {
    if (score >= 90) {
        return("A")
    } else if (score >= 80) {
        return("B")
    } else if (score >= 70) {
        return("C")
    } else {
        return("F")
    }
}
"""
    cfg = extract_r_cfg(code, "grade")

    assert cfg is not None

    # Three conditions: if, else if, else if
    # Complexity = number of decision points + 1 = 4
    assert cfg.cyclomatic_complexity == 4

    # Should have multiple branches
    assert len(cfg.edges) >= 4


# =============================================================================
# Test 4: For Loop
# =============================================================================


def test_r_cfg_for_loop():
    """For loop with (i in 1:10) syntax should create loop structure."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
sum_1_to_n <- function(n) {
    total <- 0
    for (i in 1:n) {
        total <- total + i
    }
    return(total)
}
"""
    cfg = extract_r_cfg(code, "sum_1_to_n")

    assert cfg is not None

    # For loop adds 1 decision point
    assert cfg.cyclomatic_complexity >= 2

    # Should have loop structure with back edge
    assert len(cfg.edges) >= 3


# =============================================================================
# Test 5: While Loop
# =============================================================================


def test_r_cfg_while_loop():
    """While loop should create loop structure with back edge."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
countdown <- function(n) {
    while (n > 0) {
        print(n)
        n <- n - 1
    }
}
"""
    cfg = extract_r_cfg(code, "countdown")

    assert cfg is not None

    # While loop adds 1 decision point
    assert cfg.cyclomatic_complexity >= 2

    # Should have back edge from body to header
    assert len(cfg.edges) >= 3


# =============================================================================
# Test 6: Repeat Loop (R-specific)
# =============================================================================


def test_r_cfg_repeat_loop():
    """Repeat loop should execute body before checking condition.

    This is R-SPECIFIC - repeat executes at least once unlike while/for.
    The condition is checked after body execution via break.
    """
    from tldr.cfg_extractor import extract_r_cfg

    code = """
repeat_example <- function() {
    x <- 0
    repeat {
        x <- x + 1
        if (x >= 10) {
            break
        }
    }
    return(x)
}
"""
    cfg = extract_r_cfg(code, "repeat_example")

    assert cfg is not None

    # Repeat is a loop with break as exit condition
    assert cfg.cyclomatic_complexity >= 2

    # Should have break edge handling
    break_edges = [e for e in cfg.edges if e.edge_type == "break"]
    assert len(break_edges) >= 1


# =============================================================================
# Test 7: Switch Statement (R-specific)
# =============================================================================


def test_r_cfg_switch():
    """Switch statement should create branching in CFG.

    This is R-SPECIFIC - R's switch() is different from C-style switch.
    It selects from a list of alternatives based on position or name.
    """
    from tldr.cfg_extractor import extract_r_cfg

    code = """
switch_example <- function(type) {
    result <- switch(type,
        "a" = { 1 },
        "b" = { 2 },
        "c" = { 3 },
        { 0 }  # default
    )
    return(result)
}
"""
    cfg = extract_r_cfg(code, "switch_example")

    assert cfg is not None

    # Switch with 4 cases = 4 decision points + 1 = 5 complexity
    assert cfg.cyclomatic_complexity >= 4

    # Should have multiple branches
    assert len(cfg.edges) >= 4


# =============================================================================
# Test 8: Break Statement
# =============================================================================


def test_r_cfg_break():
    """Break statement should create exit edge from loop."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
break_example <- function(values) {
    for (v in values) {
        if (v < 0) {
            break
        }
        print(v)
    }
}
"""
    cfg = extract_r_cfg(code, "break_example")

    assert cfg is not None

    # For loop + break = 2 decision points
    assert cfg.cyclomatic_complexity >= 2

    # Should have break edge
    break_edges = [e for e in cfg.edges if e.edge_type == "break"]
    assert len(break_edges) >= 1


# =============================================================================
# Test 9: Next Statement (R's continue equivalent)
# =============================================================================


def test_r_cfg_next():
    """Next statement (R's continue) should create back-edge to loop header.

    This is R-SPECIFIC - R uses 'next' instead of 'continue'.
    """
    from tldr.cfg_extractor import extract_r_cfg

    code = """
sum_odd <- function(n) {
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
    cfg = extract_r_cfg(code, "sum_odd")

    assert cfg is not None

    # For loop + if + next = 3 decision points = 4 complexity
    assert cfg.cyclomatic_complexity >= 3

    # Should have continue-like edge (next creates skip to next iteration)
    # The edge type should be "continue" or "next"
    next_edges = [e for e in cfg.edges if e.edge_type == "continue"]
    assert len(next_edges) >= 1


# =============================================================================
# Test 10: Nested Control Flow
# =============================================================================


def test_r_cfg_nested_control():
    """Nested if and loops should accumulate complexity."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
count_positive <- function(matrix) {
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
    cfg = extract_r_cfg(code, "count_positive")

    assert cfg is not None

    # Nested loops: for(outer) + for(inner) + if = 3 decision points = 4 complexity
    assert cfg.cyclomatic_complexity >= 4


# =============================================================================
# Test 11: Function Not Found Error
# =============================================================================


def test_r_cfg_function_not_found():
    """Should raise ValueError when function not found."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
exists <- function() {
    print("exists")
}
"""
    with pytest.raises(ValueError, match="not found"):
        extract_r_cfg(code, "nonexistent")


# =============================================================================
# Test: Complex R Function with Multiple Constructs
# =============================================================================


def test_r_cfg_complex_function():
    """Complex R function with multiple control flow constructs."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
process_data <- function(data, threshold) {
    # Linear code
    result <- numeric(length(data))
    count <- 0

    # For loop
    for (i in seq_along(data)) {
        # Nested if
        if (data[i] > threshold) {
            # If true, process
            if (is.na(data[i])) {
                next  # Skip NA values
            }
            result[i] <- data[i] ^ 2
            count <- count + 1
        } else if (data[i] < 0) {
            # Else if branch
            result[i] <- 0
        } else {
            # Else branch
            result[i] <- data[i]
        }
    }

    # Return result
    return(list(values=result, count=count))
}
"""
    cfg = extract_r_cfg(code, "process_data")

    assert cfg is not None
    assert cfg.function_name == "process_data"

    # Decision points: for(1) + if(1) + if nested(1) + else if(1) = 4 decision points
    # Complexity = 4 + 1 = 5
    assert cfg.cyclomatic_complexity == 5


# =============================================================================
# Test: If Without Else
# =============================================================================


def test_r_cfg_if_without_else():
    """If statement without else should handle missing branch."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
increment_if_positive <- function(x) {
    if (x > 0) {
        x <- x + 1
    }
    return(x)
}
"""
    cfg = extract_r_cfg(code, "increment_if_positive")

    assert cfg is not None

    # If without else: complexity = 2 (one decision point)
    assert cfg.cyclomatic_complexity == 2


# =============================================================================
# Test: Nested Repeat-While Pattern (R idiom)
# =============================================================================


def test_r_cfg_repeat_with_while_pattern():
    """R idiom: repeat with break is often used instead of while."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
repeat_pattern <- function(limit) {
    x <- 0
    repeat {
        x <- x + 1
        if (x >= limit) {
            break
        }
        if (x %% 3 == 0) {
            next
        }
        print(x)
    }
    return(x)
}
"""
    cfg = extract_r_cfg(code, "repeat_pattern")

    assert cfg is not None

    # Repeat + if(break) + if(next) = 3 decision points = 4 complexity
    assert cfg.cyclomatic_complexity >= 3

    # Should have both break and continue edges
    break_edges = [e for e in cfg.edges if e.edge_type == "break"]
    next_edges = [e for e in cfg.edges if e.edge_type == "continue"]
    assert len(break_edges) >= 1
    assert len(next_edges) >= 1


# =============================================================================
# Test: Switch with Expression
# =============================================================================


def test_r_cfg_switch_expression():
    """R switch used as expression with direct assignment."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
calculate <- function(op, a, b) {
    result <- switch(op,
        "add" = a + b,
        "sub" = a - b,
        "mul" = a * b,
        "div" = a / b,
        stop("Unknown operation")
    )
    return(result)
}
"""
    cfg = extract_r_cfg(code, "calculate")

    assert cfg is not None

    # Switch with 4 cases + stop = 5 decision points = 6 complexity
    assert cfg.cyclomatic_complexity >= 5


# =============================================================================
# Test: For Loop with Break and Next
# =============================================================================


def test_r_cfg_for_with_break_next():
    """For loop with break and next statements."""
    from tldr.cfg_extractor import extract_r_cfg

    code = """
process_with_limits <- function(values, max_count) {
    processed <- 0
    result <- c()
    for (v in values) {
        if (processed >= max_count) {
            break
        }
        if (is.na(v)) {
            next
        }
        result <- c(result, v)
        processed <- processed + 1
    }
    return(result)
}
"""
    cfg = extract_r_cfg(code, "process_with_limits")

    assert cfg is not None

    # For + if(break) + if(NA check) = 3 decision points = 4 complexity
    assert cfg.cyclomatic_complexity >= 3

    # Verify break and next edges exist
    break_edges = [e for e in cfg.edges if e.edge_type == "break"]
    next_edges = [e for e in cfg.edges if e.edge_type == "continue"]
    assert len(break_edges) >= 1
    assert len(next_edges) >= 1
