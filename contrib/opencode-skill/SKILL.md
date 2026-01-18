---
name: tldr-code-analysis
description: >
  Code analysis tool for semantic search, architecture discovery, and dependency
  analysis. Use when analyzing unfamiliar codebases, finding specific functions,
  understanding call graphs, detecting dead code, or performing impact analysis.
  Supports Python, TypeScript, JavaScript, Go, Rust, Java, and R.
---

# TLDR Code Analysis Skill

TLDR is a powerful code analysis tool designed to help LLMs understand and navigate codebases. It provides semantic search, architecture discovery, call graph analysis, and impact assessment capabilities.

## When to Use

Use this skill in these scenarios:

### Semantic Search for Code
When you need to find functions or code patterns based on their purpose rather than exact text:
- "Find functions that validate user input"
- "Search for database connection code"
- "Find error handling patterns"

### Understanding Codebase Architecture
When starting to work with an unfamiliar codebase:
- Get an overview of project structure
- Identify main entry points
- Understand module organization
- Discover architectural patterns

### Finding Call Graphs and Dependencies
When you need to trace code flow:
- "What calls this function?"
- "What functions does this function call?"
- "Map all dependencies of a module"
- Trace execution paths through the codebase

### Impact Analysis
Before making changes, understand what might break:
- "What would break if I modify this function?"
- Find all callers of a specific function
- Identify downstream effects of code changes
- Understand ripple effects across modules

### Dead Code Detection
When optimizing or refactoring:
- Find unused functions
- Identify unreachable code
- Discover orphaned modules
- Locate code that is never called

---

## Commands Reference

### Warm Cache
Pre-load the codebase into cache for faster subsequent queries:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr warm .
```

**When to use:** At the start of a coding session for large codebases.

### Semantic Index
Build a semantic index for natural language search:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr semantic index --lang r .
```

**When to use:** Once per project before using semantic search. Required for semantic queries.

### Semantic Search
Search code using natural language queries:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr semantic search "query" --path . --k 5
```

**When to use:** When you can't find code by exact text but know what it should do.

### Project Structure
Get an overview of project organization:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr structure . --lang r --max 20
```

**When to use:** When starting with a new codebase or understanding module layout.

### Call Graph
Generate call relationships between functions:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr calls .
```

**When to use:** When tracing code flow or finding which functions call others.

### Impact Analysis (Reverse Call Graph)
Find all functions that call a specific function:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr impact function_name . --lang r
```

**When to use:** Before modifying a function to understand its usage throughout the codebase.

### Architecture Detection
Identify architectural patterns and component relationships:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr arch . --lang r
```

**When to use:** When understanding high-level architecture or identifying design patterns.

### Extract Code Units
Extract and analyze code units from a file:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr extract file.R
```

**When to use:** When you need detailed analysis of a specific file's structure.

---

## Language Support

### Full Language Support
- **Python** (.py)
- **TypeScript** (.ts, .tsx)
- **JavaScript** (.js, .jsx)
- **Go** (.go)
- **Rust** (.rs)
- **Java** (.java)
- **R** (.R, .r)

### R-Specific Features
TLDR has enhanced support for R with additional analysis capabilities:

- **Control Flow Graph (CFG):** Detailed function control flow
- **Data Flow Graph (DFG):** Variable and data dependencies
- **Program Slicing:** Backward and forward slicing for variables
- **S7 Classes:** Full support for S7 object-oriented programming
- **Tidyverse Patterns:** Recognition of dplyr, tidyr, purrr patterns

### Specifying Language
Always use the `--lang` flag to ensure accurate analysis:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr structure . --lang r --max 20
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr semantic index --lang python .
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr impact my_function . --lang r
```

---

## Usage Pattern

Invoke TLDR directly from the GitHub fork using `uvx` without needing a local install:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr <command>
```

**Note:** The `@dev` specifies the development branch. The first run downloads dependencies (subsequent runs use cached versions).

**For R support,** use the `[r]` extra to include R-specific analysis capabilities:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev[r]' tldr <command>
```

### Working Directory
Most commands accept `.` to target the current directory (the project being analyzed):

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr structure .
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr calls .
```

### Path Specification
Replace `.` with specific paths for targeted analysis:

```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr structure ./src --lang python
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr extract ./R/utils.R
```

### Common Workflow
```bash
# 1. Warm the cache
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr warm .

# 2. Build semantic index (once)
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev[r]' tldr semantic index --lang r .

# 3. Explore structure
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev[r]' tldr structure . --lang r --max 20

# 4. Find specific code with semantic search
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev' tldr semantic search "validation function" --k 5

# 5. Trace call graph for impact analysis
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev[r]' tldr impact target_function . --lang r
```

---

## Installation

To install this skill in OpenCode, copy the `SKILL.md` file to your OpenCode skills directory:

```bash
# Create skill directory
mkdir -p ~/.config/opencode/skill/tldr-code-analysis

# Copy skill file
cp contrib/opencode-skill/SKILL.md ~/.config/opencode/skill/tldr-code-analysis/
```

---

## Integration Notes

### CLI-Based Access
TLDR runs from the command line, making it accessible to:
- **Main agents:** Can invoke directly
- **Subagents:** Can use via bash tool for nested analysis
- **Automated workflows:** Can be scripted and chained

### Cached Results
TLDR caches analysis results for performance:
- Initial analysis takes time
- Subsequent queries are fast
- Cache persists across sessions
- Use `tldr warm .` to pre-load for interactive sessions

### Semantic Search Indexing
Semantic search requires a one-time index build:
```bash
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev[r]' tldr semantic index --lang r .
```

After indexing:
- Natural language queries work efficiently
- Index is project-specific
- Rebuild index when codebase changes significantly

### Combining Commands
For comprehensive analysis, chain commands:

```bash
# Get structure and architecture
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev[r]' tldr structure . --lang r --max 30
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev[r]' tldr arch . --lang r

# Impact analysis workflow
uvx --from 'git+https://github.com/sims1253/llm-tldr@dev[r]' tldr impact function_to_modify . --lang r
```

### Best Practices
1. **Warm cache first** for interactive analysis sessions
2. **Build semantic index** before natural language searches
3. **Specify language** for accurate analysis
4. **Use impact analysis** before making changes
5. **Combine commands** for comprehensive understanding
