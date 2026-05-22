# Feature Scout Report — 2026-05-22

## Summary
| Impact | Count | Top Pick |
|--------|-------|----------|
| High (4-5) | 12 | Whole-File Fallback Size Guard |
| Medium (3) | 8 | Stale Context Detection |
| Low (1-2) | 0 | — |

Total: 17 new features identified, 3 existing items referenced.
Categories analyzed: edit-accuracy, context-management, task-decomposition, error-recovery, prompt-engineering, token-efficiency, review-quality, multi-file-reasoning

## Features

### 1. Whole-File Fallback Size Guard {IMPLEMENTED}
> Implemented: Added _WHOLE_FILE_MAX_SIZE_KB constant and size check in _whole_file_fallback(), configurable via AgentConfig.max_whole_file_size_kb. [2026-05-22]
**Category:** error-recovery
**Impact:** 4/5 — Prevents hallucination and truncation on large files, which is the most damaging Tier 3 failure mode.
**Effort:** XS — ~15 lines, size check + early return in one function, new constant in AgentConfig.
**Priority Score:** 20

**The Gap:** Cloud models handle large files by reasoning about structure; 8B models asked to regenerate a 10KB+ file often hallucinate functions or truncate mid-statement. Currently `_whole_file_fallback()` has no size limit at all.

**Proposed Approach:**
- Add `_WHOLE_FILE_MAX_SIZE_KB = 10` constant in `agent.py`
- In `_whole_file_fallback()` (line ~560), check `Path(failed_file).stat().st_size / 1024` before attempting
- If file exceeds limit, log warning and return None (skip Tier 3, let step fail gracefully)
- Add `max_whole_file_size_kb` to `AgentConfig` in `models.py` for configurability
- Log: "File too large for whole-file recovery (N KB) — skipping Tier 3"

**Files Affected:**
- `probablyfine/agent.py` — add size check in `_whole_file_fallback()`
- `probablyfine/models.py` — add field to `AgentConfig`

---

### 2. SEARCH Block Context Calculator {IMPLEMENTED}
> Implemented: Added _calculate_context_hint() to edit_parser.py that finds minimum surrounding lines needed to disambiguate multi-match SEARCH blocks. Updated validate_edits() error message and _RETRY_TEMPLATE in agent.py. [2026-05-22]
**Category:** edit-accuracy
**Impact:** 4/5 — Multi-match SEARCH blocks are the most common Tier 1 failure; telling the model exactly how many context lines are needed dramatically improves retry success.
**Effort:** S — ~40 lines, new function in edit_parser.py + update retry template in agent.py.
**Priority Score:** 16

**The Gap:** Cloud models intuitively calibrate how much surrounding context to include in a search block. 8B models either include too little (matches multiple locations) or too much (doesn't exist verbatim). Current error message is generic: "must be unique. Add more context lines."

**Proposed Approach:**
- Add `calculate_context_requirement(content, search_text, match_count)` to `edit_parser.py`
- Scan backwards/forwards from each match to find minimum unique prefix/suffix
- Return `(min_lines_needed, anchor_excerpt)` — e.g., "Add 5 lines above to disambiguate"
- Update `_RETRY_TEMPLATE` in `agent.py` to include: "SEARCH matched N locations. Include N lines of surrounding code to make it unique."
- Pass context requirement through `validate_edits()` error tuple

**Files Affected:**
- `probablyfine/edit_parser.py` — add `calculate_context_requirement()`, enhance error detail in `validate_edits()`
- `probablyfine/agent.py` — update `_RETRY_TEMPLATE` to include context guidance

---

### 3. Fuzzy Anchor-Based Search {IMPLEMENTED}
> Implemented: Added _fuzzy_anchor_replace() to edit_parser.py — extracts first/last non-blank non-comment lines as anchors, finds them in file content, replaces the region between. Inserted into _apply_single_edit() fallback chain between indent-fuzzy and failure. [2026-05-22]
**Category:** edit-accuracy
**Impact:** 4/5 — Catches the gap between indent-fuzzy matching (implemented) and whole-file fallback. Many failures involve correct core logic with drifted surrounding context.
**Effort:** S — ~40 lines, new function in edit_parser.py inserted into existing fallback chain.
**Priority Score:** 16

**The Gap:** Cloud models regenerate correct SEARCH blocks even when comments or blank lines shifted. 8B models generate SEARCH blocks where the core edited lines are correct but anchor lines (first/last non-blank lines) have drifted. Current fuzzy matching only handles whitespace differences.

**Proposed Approach:**
- Add `_fuzzy_anchor_replace(content, search, replace)` to `edit_parser.py`
- Extract first and last non-blank, non-comment lines of SEARCH as anchors
- Find those anchors in file content (allowing whitespace-fuzzy match)
- Replace all content between found anchors with replacement text
- Insert into `_apply_single_edit()` fallback chain between indent-fuzzy and failure
- Log "Applied anchor-fuzzy match" when used

**Files Affected:**
- `probablyfine/edit_parser.py` — add `_fuzzy_anchor_replace()`, insert into `_apply_single_edit()` fallback chain

---

### 4. Targeted Retry Context by Error Type {IMPLEMENTED}
> Implemented: Added _classify_edit_error() to categorize failures as multi_match/not_found/generic. Enhanced _get_nearby_content() with error-type branching: multi_match shows all match locations, not_found uses difflib for closest match. Updated _RETRY_TEMPLATE with per-type guidance. [2026-05-22]
**Category:** error-recovery
**Impact:** 4/5 — Generic retry prompts waste the model's second chance. Telling the model exactly what type of error it made enables targeted self-correction.
**Effort:** S — ~50 lines in agent.py, modifying `_get_nearby_content()` and `_retry_with_error_context()`.
**Priority Score:** 16

**The Gap:** Cloud models understand their own errors from context. 8B models given a retry prompt with just "error + nearby content" often repeat the same mistake. Categorized error context (multi-match → show all matches; hallucination → show file outline) enables targeted correction.

**Proposed Approach:**
- Modify `_get_nearby_content()` to accept `error_type` parameter
- For "multi_match": show all match locations with line numbers
- For "not_found" with close match: show closest match via difflib.get_close_matches
- For "hallucination" (non-existent names): show file outline (function/class names via ast)
- Update `_retry_with_error_context()` to pass error_type from validation

**Files Affected:**
- `probablyfine/agent.py` — modify `_get_nearby_content()` signature, add error-type branching, update `_retry_with_error_context()`

---

### 5. Symbol Index for Decomposer {IMPLEMENTED}
> Implemented: Added _build_codebase_summary() to interpreter.py — uses ast.parse to extract top-level function/class names per Python file, capped at 50 symbols. Appended to file_summary in _decompose_and_parse() so the decomposer sees real symbol names. [2026-05-22]
**Category:** task-decomposition
**Impact:** 4/5 — Decomposer currently gets file names + line counts but no code structure. Adding function/class names reduces hallucinated step targets from ~35% to <10%.
**Effort:** S — ~50 lines, new function in interpreter.py using ast.walk.
**Priority Score:** 16

**The Gap:** Cloud models analyze code structure during planning. The probablyfine decomposer gets file summaries with only names and line counts but no symbols. The model hallucinates function names or proposes edits to non-existent targets.

**Proposed Approach:**
- Add `_build_codebase_summary(file_context, max_symbols=50)` to `interpreter.py`
- Use `ast.parse` + `ast.walk` to extract function/class names per file
- Append to decompose prompt: "Available symbols:\n  file.py: func1, func2, ClassName"
- Cap at 50 symbols total to stay within token budget
- Call in `_decompose_and_parse()` alongside existing `_build_file_summary()`

**Files Affected:**
- `probablyfine/interpreter.py` — add `_build_codebase_summary()`, inject into `DECOMPOSE_PROMPT` formatting

---

### 6. Classifier/Decomposer Few-Shot Examples {IMPLEMENTED}
> Implemented: Added 2 few-shot examples to CLASSIFY_PROMPT (bug_fix complexity=1 + refactor complexity=3) and 1 few-shot example to DECOMPOSE_PROMPT (3-step feature edit). All inline in interpreter.py, compact to respect token budgets. [2026-05-22]
**Category:** prompt-engineering
**Impact:** 4/5 — Agent prompts already have few-shot examples, but classifier and decomposer have none. Adding 2-3 examples reduces classification errors and decomposition hallucinations.
**Effort:** S — ~40 lines of prompt text additions across 2 files.
**Priority Score:** 16

**The Gap:** Cloud models use in-context learning naturally. The classifier and decomposer in interpreter.py have detailed instructions but zero examples. 8B models benefit disproportionately from few-shot examples compared to instruction-only prompts.

**Proposed Approach:**
- Add `_CLASSIFY_EXAMPLES` with 2 examples: simple task (bug_fix, complexity 1) and complex task (refactor, complexity 3)
- Add `_DECOMPOSE_EXAMPLES` with 1 example: 3-step edit task showing correct JSON structure
- Inject conditionally via `get_prompt_suffix()` in `ollama_utils.py`
- Keep examples compact (< 200 tokens each) to respect 8B context limits

**Files Affected:**
- `probablyfine/interpreter.py` — add example constants, inject into prompts
- `probablyfine/ollama_utils.py` — extend `get_prompt_suffix()` to handle examples phase

---

### 7. Pre-Checker Sanity Filter {IMPLEMENTED}
> Implemented: Added quick_sanity_check() to checker.py — ast.parse on changed Python files to catch syntax errors in O(1). Called before LLM checker in _run_checker_loop() in reflection.py; returns immediate FAIL with structured issues. [2026-05-22]
**Category:** review-quality
**Impact:** 4/5 — Catches obvious bugs (syntax errors, undefined references) in O(1) before invoking the expensive checker LLM. Saves 30-60s per reflection iteration on trivially broken diffs.
**Effort:** S — ~40 lines, new function called before `run_checker()`.
**Priority Score:** 16

**The Gap:** Cloud models pre-filter trivial issues internally. Currently every checker invocation is full-price even when the diff has obvious Python syntax errors. A quick `ast.parse` + reference check could catch 20-30% of failures instantly.

**Proposed Approach:**
- Add `_quick_sanity_check(diff, changed_files)` to `checker.py`
- For Python files in diff: attempt `ast.parse` on the post-edit content — if SyntaxError, return immediate FAIL
- Check for undefined name references: if diff adds a call to `foo()` but no `def foo` or `import foo` exists
- Call before `run_checker()` in `_run_checker_loop()` — if sanity check fails, skip LLM and return structured CheckerResult
- Log: "Pre-checker caught syntax error — skipping LLM review"

**Files Affected:**
- `probablyfine/checker.py` — add `_quick_sanity_check()` function
- `probablyfine/reflection.py` — call before `run_checker()` in `_run_checker_loop()`

---

### 8. Intelligent File Content Sampling {EXISTING: next-improvements #17}
**Category:** context-management
**Impact:** 5/5 — Reduces context size 30-50% for large codebases by sending function signatures + docstrings instead of full bodies for non-targeted files.
**Effort:** M — ~100 lines, new AST-based sampler + integration into `_format_file_contents()`.
**Priority Score:** 15

**The Gap:** Cloud models reason about code structure from summaries. probablyfine sends entire file contents via `_format_file_contents()`, consuming tokens on irrelevant function bodies. A 10KB file with 20 functions wastes tokens on 18 functions the model doesn't need.

**Proposed Approach:**
- Add `_sample_python_file(fpath, max_bytes=2000)` to `agent.py` using `ast.parse`
- Extract: imports, class/function signatures + first-line docstrings, collapse bodies to `...`
- For non-Python files: truncate to `max_bytes`
- In `_format_file_contents()`, use sampling for "context" files (not the primary edit target)
- Primary edit target files still get full content
- Configurable via `AgentConfig.context_sampling` bool

**Files Affected:**
- `probablyfine/agent.py` — add `_sample_python_file()`, modify `_format_file_contents()`
- `probablyfine/models.py` — add `context_sampling` to `AgentConfig`

---

### 9. Stale Context Detection {IMPLEMENTED}
> Implemented: Added _mtimes dict, update_mtime(), needs_refresh(), stale_files() to FileContext in context.py. Added module-level _file_mtimes cache in agent.py _format_file_contents() — detects and logs when files have been modified between steps. [2026-05-22]
**Category:** context-management
**Impact:** 3/5 — Catches subtle bugs when step N edits a file that step N+1's context was built from. Important for multi-step plans.
**Effort:** XS — ~20 lines, mtime tracking in FileContext + check in `_format_file_contents()`.
**Priority Score:** 15

**The Gap:** Cloud models re-read files naturally between steps. probablyfine reads files once at step start and only refreshes when explicitly triggered. If step 2 edits `auth.py` and step 3 reads stale `auth.py` content, step 3 may generate incorrect edits.

**Proposed Approach:**
- Add `_mtimes: dict[str, float]` to `FileContext` class in `context.py`
- Add `needs_refresh(fpath)` method: compare current mtime to stored mtime
- Add `update_mtime(fpath)` method: record after reading
- In `_format_file_contents()`, log warning if any file needs refresh
- In `execute_plan()`, automatically re-read stale files before each step

**Files Affected:**
- `probablyfine/context.py` — add mtime tracking to `FileContext`
- `probablyfine/agent.py` — check staleness in `_format_file_contents()`

---

### 10. Dynamic Step Cap by Complexity {IMPLEMENTED}
> Implemented: Added _get_step_budget(complexity, task_len) — complexity 1→3 steps, 2→6, 3→10, plus bonus for long task descriptions. Updated DECOMPOSE_PROMPT with {max_steps} placeholder, threaded complexity through _decompose_and_parse() and _decompose_task(). [2026-05-22]
**Category:** task-decomposition
**Impact:** 3/5 — Complex tasks (complexity=3) get truncated at 6 steps. Allowing 8-10 steps for detailed tasks improves plan quality.
**Effort:** XS — ~15 lines, new function + update to cap logic in interpreter.py.
**Priority Score:** 15

**The Gap:** Cloud models generate plans of appropriate length. probablyfine caps at `MAX_DECOMPOSITION_STEPS = 6` regardless of task complexity. Simple tasks need 1-2 steps; complex refactors need 8-10.

**Proposed Approach:**
- Add `_get_step_budget(complexity, task_len)` to `interpreter.py`
- Complexity 1 → max 3 steps, complexity 2 → max 6, complexity 3 → max 10
- Bonus: +1 step per 100 chars of task description, capped at 10
- Update `DECOMPOSE_PROMPT` to include dynamic `{max_steps}` placeholder
- Update `_decompose_and_parse()` to use dynamic cap instead of constant

**Files Affected:**
- `probablyfine/interpreter.py` — add `_get_step_budget()`, update prompt and cap logic

---

### 11. Edit Block Ordering (Bottom-Up Apply) {IMPLEMENTED}
> Implemented: Added _sort_edits_bottom_up() to edit_parser.py — groups edits by file, sorts each group by estimated line position descending (line-anchored or SEARCH text position). Called at start of apply_edits_atomic(). [2026-05-22]
**Category:** edit-accuracy
**Impact:** 3/5 — Applying multiple edits to the same file top-down shifts line numbers. Bottom-up apply eliminates this class of failures.
**Effort:** S — ~30 lines, new sort function in edit_parser.py.
**Priority Score:** 12

**The Gap:** Cloud models generate edits in safe order. 8B models list edits in narrative order (top-to-bottom), but applying top-to-bottom shifts line numbers for later edits.

**Proposed Approach:**
- Add `_sort_edits_bottom_up(edits)` to `edit_parser.py`
- For line-anchored edits: sort by `line_end` descending within each file
- For SEARCH/REPLACE: estimate position by finding search text in file, sort descending
- Call at the start of `apply_edits_atomic()` before the apply loop

**Files Affected:**
- `probablyfine/edit_parser.py` — add `_sort_edits_bottom_up()`, call in `apply_edits_atomic()`

---

### 12. Structured Edit Error Classification {IMPLEMENTED}
> Implemented: Added EDIT_ERR_* constants to models.py. Updated validate_edits() in edit_parser.py to return 3-tuples (edit, msg, error_type). Updated agent.py to unpack error_type and pass directly to _retry_with_error_context(), removing the string-based _classify_edit_error(). [2026-05-22]
**Category:** edit-accuracy
**Impact:** 4/5 — Classifying errors as hallucination/multi-match/indent-mismatch enables targeted recovery strategies instead of generic retries.
**Effort:** M — ~100 lines across 3 files (new dataclass, classification logic, retry integration).
**Priority Score:** 12

**The Gap:** Cloud models understand error types from context. probablyfine treats all SEARCH validation failures identically. Knowing the error TYPE enables the retry prompt to give targeted guidance.

**Proposed Approach:**
- Add `EditErrorType` enum to `models.py`: `NOT_FOUND`, `MULTI_MATCH`, `INDENT_MISMATCH`, `HALLUCINATION`
- Add `EditValidationError` dataclass with error_type, match_count, closest_match, suggestion
- In `validate_edits()`, classify each error using difflib and ast
- Return structured errors instead of `(edit, error_string)` tuples

**Files Affected:**
- `probablyfine/models.py` — add `EditErrorType` enum, `EditValidationError` dataclass
- `probablyfine/edit_parser.py` — refactor `validate_edits()` return type
- `probablyfine/agent.py` — consume structured errors in `_execute_edit()`

---

### 13. Checker Failure Mode Categorization {IMPLEMENTED}
> Implemented: Added CHECKER_* failure mode constants and failure_mode field to CheckerResult in models.py. Tagged all failure paths in checker.py (hang, OOM, timeout, empty, parse_fail). Added failure_mode reaction in reflection.py _run_checker_loop() — hang/OOM skips remaining iterations. [2026-05-22]
**Category:** error-recovery
**Impact:** 3/5 — Different checker failures (hang, OOM, timeout) should trigger different recovery strategies instead of uniform "PASS with 0 confidence."
**Effort:** S — ~50 lines across 3 files.
**Priority Score:** 12

**The Gap:** When probablyfine's checker hangs, OOMs, or times out, it uniformly returns a lenient PASS. But hang → skip remaining iterations; OOM → reduce diff size; timeout → model is slow, not wrong.

**Proposed Approach:**
- Add `CheckerFailureMode` enum to `models.py`: `HANG`, `OOM`, `TIMEOUT`, `INVALID_JSON`
- Add `failure_mode` field to `CheckerResult`
- In `run_checker()`, categorize exceptions
- In `_run_checker_loop()`, react to failure_mode appropriately

**Files Affected:**
- `probablyfine/models.py` — add enum, update `CheckerResult`
- `probablyfine/checker.py` — categorize failures
- `probablyfine/reflection.py` — react to failure_mode

---

### 14. Import Chain Following for Auto-Selection {IMPLEMENTED}
> Implemented: Added _extract_imports() (ast-based) and _follow_import_chain() (BFS depth=1, cap=20 files) to file_selector.py. Builds module-to-file mapping from git files, follows imports from seed files. Integrated into select_files() after keyword+LLM merge. [2026-05-22]
**Category:** context-management
**Impact:** 4/5 — File selector misses imported files. Following import chains improves recall from ~65% to 85%+.
**Effort:** M — ~80 lines, ast-based import extraction + BFS traversal.
**Priority Score:** 12

**The Gap:** Cloud models reason about module dependencies. probablyfine's file selector uses keyword matching + LLM guess but never follows import chains.

**Proposed Approach:**
- Add `_follow_import_chain(git_files, seed_files, max_depth=1)` to `file_selector.py`
- Use `ast.parse` + `ast.walk` to find ImportFrom and Import nodes
- Map module paths back to repo files via substring matching
- BFS to depth 1, cap at 20 files total

**Files Affected:**
- `probablyfine/file_selector.py` — add `_follow_import_chain()`, integrate into `select_files()`

---

### 15. Complexity-Aware /no_think Scheduling {IMPLEMENTED}
> Implemented: DECOMPOSE_PROMPT uses {thinking_suffix} placeholder; _decompose_task() conditionally omits /no_think and increases num_predict to 3000 when complexity >= 3. [2026-05-22]
**Category:** prompt-engineering
**Impact:** 3/5 — Allowing thinking tokens for complexity=3 tasks improves decomposition quality at the cost of some tokens.
**Effort:** S — ~30 lines, conditional suffix logic in interpreter.py.
**Priority Score:** 12

**The Gap:** qwen3:8b's thinking mode is disabled via `/no_think` for all structured output. But complex tasks benefit from thinking tokens.

**Proposed Approach:**
- Keep `/no_think` for classification (always structured)
- Conditionally omit `/no_think` for decompose when complexity=3
- Increase `DECOMPOSE_NUM_PREDICT` from 2000 to 3000 when thinking is allowed
- Strip thinking tags via `strip_think_tags()` before JSON parsing

**Files Affected:**
- `probablyfine/interpreter.py` — conditional `/no_think` in `_decompose_task()`, dynamic num_predict

---

### 16. Import Graph for Pre-Execution Planning {IMPLEMENTED}
> Implemented: Added _build_import_graph() using ast-based import analysis. Integrated into execute_plan() — graph built before step execution, used by _should_replan() (triggers replan when file has >2 dependents) and _refresh_step_files() (adds dependent files to context). [2026-05-22]
**Category:** multi-file-reasoning
**Impact:** 4/5 — Cross-file consistency check is post-hoc only. Pre-execution import graph enables proactive dependency-aware step context.
**Effort:** M — ~80 lines, graph builder + integration into execute_plan().
**Priority Score:** 12

**The Gap:** `_verify_cross_file_consistency()` only runs AFTER all edits. If step 2 renames a function in module A, step 3 should already know about modules B and C that import from A.

**Proposed Approach:**
- Add `_build_import_graph(files)` to `agent.py` — returns file→imports mapping
- Call at start of `execute_plan()` before executing steps
- When step targets file A, check graph for dependents, add to step context
- Inform `_should_replan()`: replan if edited file has >2 dependents

**Files Affected:**
- `probablyfine/agent.py` — add `_build_import_graph()`, integrate into `execute_plan()` and `_should_replan()`

---

### 17. Dynamic Token Budget by Context Size {IMPLEMENTED}
> Implemented: Added _measure_context() and scaled _get_step_budget() by context_bytes and file_count (0.5x–2.0x). Updated all 3 call sites in _execute_edit and _execute_explain. [2026-05-22]
**Category:** token-efficiency
**Impact:** 3/5 — Static num_predict values don't account for context size. Scaling by context prevents both waste and truncation.
**Effort:** S — ~30 lines, modify `_get_step_budget()` to accept context parameters.
**Priority Score:** 12

**The Gap:** Cloud models adjust output length based on input complexity. probablyfine assigns fixed token budgets per action type regardless of context size or file count.

**Proposed Approach:**
- Modify `_get_step_budget(action, context_size=0, file_count=1)` in `agent.py`
- Scale: `1.0 + (context_size / 48000) * 0.5 + (min(file_count, 5) / 5) * 0.3`, clamped 0.5x–2.0x
- Pass context size from `_execute_edit()` and `_execute_explain()`
- Log adjusted budget

**Files Affected:**
- `probablyfine/agent.py` — modify `_get_step_budget()` signature and callers

---

### 18. Misbehavior Observer: Oscillation Detection {EXISTING: next-improvements #4}
**Category:** error-recovery
**Impact:** 3/5 — Current observer only detects exact reasoning loops. Oscillation detection catches fail→ok→fail→ok patterns.
**Effort:** S — ~40 lines, extend `_MisbehaviorObserver` class.
**Priority Score:** 12

**The Gap:** The current observer (agent.py) only catches exact repetitions. It misses alternating patterns where the model flip-flops between approaches.

**Proposed Approach:**
- Add `check_oscillation()` — track success/failure in 6-item window, detect alternating pattern
- Add `check_escalation()` — flag if each retry error is longer than previous
- Call both in `execute_plan()` step loop
- On oscillation: break plan; on escalation: log warning

**Files Affected:**
- `probablyfine/agent.py` — extend `_MisbehaviorObserver` class, add calls in `execute_plan()`

---

### 19. Unified Diff Format Support {IMPLEMENTED}
> Implemented: Added _UNIFIED_DIFF_RE, _HUNK_HEADER_RE regexes and _parse_unified_diff() to edit_parser.py. Integrated as fourth format in parse_edits(). Added unified diff documentation to AGENT_SYSTEM_PROMPT. [2026-05-22]
**Category:** edit-accuracy
**Impact:** 3/5 — Alternative edit format for cases where SEARCH/REPLACE fails. Some 8B models generate unified diffs more reliably.
**Effort:** M — ~120 lines, new regex + parser + prompt update.
**Priority Score:** 9

**The Gap:** probablyfine only supports SEARCH/REPLACE, CONTENT/END, and line-anchored formats. Some 8B models produce correct unified diffs more reliably because they're trained on git diff output.

**Proposed Approach:**
- Add `_UNIFIED_DIFF_RE` regex for `@@` hunk headers to `edit_parser.py`
- Add `_parse_unified_diff(match)` — convert hunk to FileEdit with line_start/line_end
- Insert into `parse_edits()` as fourth format check
- Update `AGENT_SYSTEM_PROMPT` to document format as optional

**Files Affected:**
- `probablyfine/edit_parser.py` — add regex, parser, integrate into `parse_edits()`
- `probablyfine/agent.py` — update system prompt

---

### 20. Reusable Plan Templates {IMPLEMENTED}
> Implemented: Added _PLAN_TEMPLATES list with 6 patterns (add_import, rename_symbol, add_config, simple_bugfix, add_function, add_constant) and _match_template() keyword matcher. Integrated as Phase 3b in interpret_task(), before LLM decompose. [2026-05-22]
**Category:** task-decomposition
**Impact:** 3/5 — Common patterns (30-50% of tasks) could skip the 60s LLM decompose call by using keyword-matched templates.
**Effort:** M — ~100 lines, template dict + keyword matching.
**Priority Score:** 9

**The Gap:** Cloud models decompose instantly. probablyfine's decompose phase takes 30-60s per task via LLM. Common patterns could use pre-built templates.

**Proposed Approach:**
- Add `_PLAN_TEMPLATES` dict to `interpreter.py` with 5-6 patterns: add_endpoint, fix_import, rename_symbol, add_config, simple_bugfix
- Add `_match_template(task, intent)` — keyword heuristic
- In `interpret_task()`, check templates before LLM decompose
- Templates use placeholder files; file selector fills them in
- Fallback to LLM if no match or template errors

**Files Affected:**
- `probablyfine/interpreter.py` — add templates dict, matching function, integrate into `interpret_task()`

---

## Existing Items Referenced
| Source File | Item # | Title | Status |
|---|---|---|---|
| next-improvements.md | 17 | Incremental Context Compression | OPEN |
| next-improvements.md | 4 | Misbehavior Observer | OPEN |
| next-improvements.md | 2 | Destructive Edit Protection | OPEN |

## Priority Matrix
| Score | Feature | Impact | Effort | Category |
|-------|---------|--------|--------|----------|
| 20 | Whole-File Fallback Size Guard | 4 | XS | error-recovery |
| 16 | SEARCH Block Context Calculator | 4 | S | edit-accuracy |
| 16 | Fuzzy Anchor-Based Search | 4 | S | edit-accuracy |
| 16 | Targeted Retry Context by Error Type | 4 | S | error-recovery |
| 16 | Symbol Index for Decomposer | 4 | S | task-decomposition |
| 16 | Classifier/Decomposer Few-Shot Examples | 4 | S | prompt-engineering |
| 16 | Pre-Checker Sanity Filter | 4 | S | review-quality |
| 15 | Intelligent File Content Sampling | 5 | M | context-management |
| 15 | Stale Context Detection | 3 | XS | context-management |
| 15 | Dynamic Step Cap by Complexity | 3 | XS | task-decomposition |
| 12 | Edit Block Ordering (Bottom-Up) | 3 | S | edit-accuracy |
| 12 | Structured Edit Error Classification | 4 | M | edit-accuracy |
| 12 | Checker Failure Mode Categorization | 3 | S | error-recovery |
| 12 | Import Chain Following | 4 | M | context-management |
| 12 | Complexity-Aware /no_think Scheduling | 3 | S | prompt-engineering |
| 12 | Import Graph for Pre-Execution Planning | 4 | M | multi-file-reasoning |
| 12 | Dynamic Token Budget by Context Size | 3 | S | token-efficiency |
| 12 | Misbehavior Observer: Oscillation | 3 | S | error-recovery |
| 9 | Unified Diff Format Support | 3 | M | edit-accuracy |
| 9 | Reusable Plan Templates | 3 | M | task-decomposition |
