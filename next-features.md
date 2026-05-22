# Feature Scout Report — 2026-05-22

## Summary
| Impact | Count | Top Pick |
|--------|-------|----------|
| High (4-5) | 8 | Few-Shot Edit Examples in System Prompt |
| Medium (3) | 7 | Hallucination Validator Enhancement |
| Low (1-2) | 3 | Dynamic num_predict by Step Type |

Total: 18 new features identified, 4 existing items referenced.
Categories analyzed: edit-accuracy, context-management, task-decomposition, error-recovery, prompt-engineering, token-efficiency, review-quality, multi-file-reasoning

## Features

### 1. Few-Shot Edit Examples in System Prompt {IMPLEMENTED}
> Implemented: Added 3 few-shot examples (edit, new file, multi-block) to AGENT_SYSTEM_PROMPT in agent.py. [2026-05-22]

**Category:** prompt-engineering
**Impact:** 5/5 — 8B models are dramatically more format-compliant when given concrete examples; logs show frequent SEARCH block mismatches that examples would prevent.
**Effort:** XS — Add 2-3 canonical SEARCH/REPLACE examples to the agent system prompt in `agent.py`, ~15 lines.
**Priority Score:** 25

**The Gap:** Cloud models internalize edit formats from their training data and reliably produce well-formed SEARCH/REPLACE blocks. 8B models frequently emit malformed blocks — wrong delimiters, non-unique search text, or inverted search/replace sections. Few-shot examples in the system prompt are the single most effective way to compensate for smaller model capacity.

**Proposed Approach:**
- Add 2-3 example SEARCH/REPLACE blocks to the system prompt in `agent.py` showing exact format with `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` delimiters
- Include one example of a CONTENT/END block for new files
- Include one negative example showing a common mistake (e.g., using `---` instead of `=======`)
- Place examples after the format description but before the task injection

**Files Affected:**
- `probablyfine/agent.py` — Add few-shot examples to system prompt constant

---

### 2. Zero-Token Detection and Auto-Retry {IMPLEMENTED}
> Implemented: Reduced _ZERO_TOKEN_ABORT_S from 30→15s in checker.py, added auto-retry with halved num_ctx on zero-token stall. Added _ZERO_TOKEN_ABORT_S=15 to agent.py streaming loop, raises _HangDetected for existing recovery. [2026-05-22]

**Category:** error-recovery
**Impact:** 5/5 — Checker logs show 5/22 sessions (23%) producing zero tokens before the 120s timeout; this is the single largest source of wasted time.
**Effort:** S — Add early detection in `_run_checker_stream()` and agent streaming, ~40 lines across 2 files.
**Priority Score:** 20

**The Gap:** Cloud APIs never stall silently — they either respond or return an error within seconds. Local Ollama models can stall indefinitely due to VRAM pressure, KV cache allocation failure, or model loading delays. The current 30s zero-token abort in `checker.py` helps but the agent has no equivalent, and 30s is still too long when the model is clearly stalled.

**Proposed Approach:**
- Reduce `_ZERO_TOKEN_ABORT_S` in `checker.py` from 30 to 15 seconds
- Add equivalent zero-token early abort to agent streaming in `agent.py`
- On zero-token abort, automatically retry once with reduced `num_ctx` (halved) to relieve VRAM pressure
- Log the stall event with VRAM snapshot from DRM for debugging
- If retry also stalls, fall through to existing graceful degradation

**Files Affected:**
- `probablyfine/checker.py` — Reduce abort threshold, add auto-retry with reduced context
- `probablyfine/agent.py` — Add zero-token detection to streaming loop

---

### 3. Indentation-Aware Fuzzy Matching {IMPLEMENTED}
> Implemented: Added _detect_indent() and _indent_fuzzy_replace() to edit_parser.py. Integrated as Tier 2.5 fallback in both validate_edits() and _apply_single_edit() — strips leading whitespace for matching, then re-applies the file's indentation pattern to the replacement. [2026-05-22]

**Category:** edit-accuracy
**Impact:** 4/5 — Edit parser logs show repeated Tier 2 failures where search text differs only in indentation; this is the most common edit failure mode for 8B models.
**Effort:** S — ~40 lines modifying `_fuzzy_replace()` and `_normalize_whitespace()` in `edit_parser.py`.
**Priority Score:** 16

**The Gap:** Cloud models produce edits with correct indentation because they can precisely track whitespace from file content in their large context windows. 8B models frequently get indentation wrong — off by one tab, mixing spaces and tabs, or using the wrong indent level after copy-pasting from context. The current fuzzy matcher only normalizes trailing whitespace, not leading whitespace.

**Proposed Approach:**
- Add an indentation-normalizing comparison mode to `_fuzzy_replace()` in `edit_parser.py` that strips leading whitespace before matching, then re-applies the original file's indentation pattern to the replacement
- Implement as a Tier 2.5 fallback: try exact match → try whitespace-normalized match → try indent-normalized match → Tier 3 whole-file
- Detect the file's indentation style (tabs vs. spaces, indent width) from surrounding context
- Preserve the replacement's relative indentation structure while adjusting the base indent level

**Files Affected:**
- `probablyfine/edit_parser.py` — Enhance `_fuzzy_replace()` and add indent detection helper

---

### 4. File-Aware Decomposition {IMPLEMENTED}
> Implemented: Added _build_file_summary() that produces "path (N lines)" entries. Updated DECOMPOSE_PROMPT to require files arrays. Changed _decompose_and_parse() to inject file summary instead of bare path list. [2026-05-22]

**Category:** task-decomposition
**Impact:** 4/5 — Interpreter logs show most decomposed plans return empty `"files": []` arrays, forcing the agent to guess which files to edit; cloud models always know their targets.
**Effort:** S — Modify `DECOMPOSE_PROMPT` in `interpreter.py`, inject file tree summary, ~30 lines.
**Priority Score:** 16

**The Gap:** Cloud models with 128k+ context windows can see the entire project structure and produce plans that reference specific files. 8B models working from a generic prompt with `"(no files specified)"` as file context produce generic plans with empty file lists, making downstream execution imprecise.

**Proposed Approach:**
- In `_decompose_and_parse()`, build a compact file tree summary (path + line count) from `file_context` and inject it into `DECOMPOSE_PROMPT`
- Format as `"Available files:\n  probablyfine/agent.py (850 lines)\n  probablyfine/checker.py (353 lines)\n..."`
- Add an explicit instruction: "You MUST populate the files array for every edit/create/delete step using paths from the list above"
- Cap the file tree to fit within token budget (~500 tokens)

**Files Affected:**
- `probablyfine/interpreter.py` — Enhance `DECOMPOSE_PROMPT` template and `_decompose_and_parse()` to inject file metadata

---

### 5. Cascading Failure Prevention {IMPLEMENTED}
> Implemented: Added failed_step_ids tracking in execute_plan(). Steps with failed dependencies are skipped with status="skipped". Failed steps no longer break the loop — dependents are skipped and independent steps continue. [2026-05-22]

**Category:** error-recovery
**Impact:** 4/5 — Agent logs show steps executing after previous dependent steps failed (0/2 edit match rate followed by more edits on the same file), causing cascading corruption.
**Effort:** S — Add failure gates in `agent.py` `execute_plan()`, ~30 lines.
**Priority Score:** 16

**The Gap:** Cloud agents track step success/failure and skip dependent steps when a prerequisite fails, preserving a clean state. The current agent executes all steps sequentially regardless of previous failures, which means a failed edit in step 2 corrupts the context for steps 3-6.

**Proposed Approach:**
- In `execute_plan()` in `agent.py`, track each step's `StepResult.status`
- Before executing a step, check its `depends_on` list against completed step statuses
- If any dependency failed, skip the step with `status="skipped"` and `error="dependency N failed"`
- Report partial results (which steps succeeded, which were skipped) in the `AgentResult`
- Log the skip chain for debugging

**Files Affected:**
- `probablyfine/agent.py` — Add dependency checking in `execute_plan()` loop

---

### 6. Hallucination Validator Enhancement {IMPLEMENTED}
> Implemented: Added fuzzy basename recovery and README.md filtering to _parse_file_list() in file_selector.py. Near-miss paths like "promotion.html" now resolve to "pages/promotion.html" when basename is unique. README.md dropped unless task mentions docs. [2026-05-22]

**Category:** context-management
**Impact:** 3/5 — File selector logs show ~30% of LLM selections include at least one hallucinated path, with multiple cases of all paths being hallucinated and falling through to keyword backup.
**Effort:** XS — Add fuzzy basename matching to `_parse_file_list()` in `file_selector.py`, ~15 lines.
**Priority Score:** 15

**The Gap:** Cloud models with large context windows rarely hallucinate file paths because they can see the full file listing. 8B models frequently invent plausible-but-wrong paths (e.g., `promotion.html` when the tracked file is `pages/promotion.html`, or always including `README.md`). The current validator does strict equality matching and drops near-misses entirely.

**Proposed Approach:**
- In `_parse_file_list()` in `file_selector.py`, before dropping a path as hallucinated, try basename matching against the `git_files` set
- If exactly one git file matches the hallucinated basename, substitute it
- If multiple matches, log a warning and drop (ambiguous)
- Add special handling for `README.md` — always drop it unless the task mentions documentation (it's hallucinated in 80% of selections)

**Files Affected:**
- `probablyfine/file_selector.py` — Enhance `_parse_file_list()` with fuzzy matching fallback

---

### 7. Decomposition Timeout Resilience {IMPLEMENTED}
> Implemented: Added streaming=True mode to _call_llm() that captures tokens as they arrive. On timeout, partial response is salvaged via _repair_truncated_json(). _decompose_task() now uses streaming mode. Non-decompose phases unchanged. [2026-05-22]

**Category:** task-decomposition
**Impact:** 3/5 — Interpreter logs show 5 decomposition timeouts producing single-step fallback plans; partial plans would be significantly better.
**Effort:** XS — Add mid-timeout JSON salvage in `_decompose_and_parse()`, ~15 lines in `interpreter.py`.
**Priority Score:** 15

**The Gap:** Cloud models respond within seconds. 8B models on constrained hardware frequently exceed the 60s decomposition timeout, especially with thinking tokens enabled. The current fallback is a generic single-step plan that discards all the model's partial output. The `_repair_truncated_json()` infrastructure already exists but isn't applied to mid-timeout situations.

**Proposed Approach:**
- In `_call_llm()` in `interpreter.py`, when a timeout occurs during streaming, capture the partial response buffer instead of discarding it
- Apply `_repair_truncated_json()` to the partial response to salvage any complete steps
- If at least 1 valid step is recovered, use it as the plan instead of falling back to single-step
- Log "Salvaged N steps from timed-out decomposition" for observability

**Files Affected:**
- `probablyfine/interpreter.py` — Modify `_call_llm()` to capture partial streaming responses on timeout

---

### 8. Negative Examples for Checker {IMPLEMENTED}
> Implemented: Added 2 negative examples (style/docs) and 1 positive example (index bounds crash) to CHECKER_SYSTEM_PROMPT in checker.py. [2026-05-22]

**Category:** prompt-engineering
**Impact:** 3/5 — Checker occasionally flags style issues despite explicit instructions not to; concrete negative examples would reinforce the boundary.
**Effort:** XS — Add 1-2 negative examples to `CHECKER_SYSTEM_PROMPT` in `checker.py`, ~10 lines.
**Priority Score:** 15

**The Gap:** Cloud models reliably follow negative constraints ("do NOT flag style issues") from natural language instructions alone. 8B models benefit significantly from concrete negative examples — showing them "this is NOT an issue, do not flag it" alongside the real issues they should catch.

**Proposed Approach:**
- Add a "NOT an issue" section to `CHECKER_SYSTEM_PROMPT` in `checker.py` with 1-2 examples:
  - Example: "Missing docstring on a function" → NOT an issue
  - Example: "Variable named `x` instead of `descriptive_name`" → NOT an issue
- Add a "IS an issue" example showing a real bug for contrast
- Keep examples minimal to avoid consuming checker token budget

**Files Affected:**
- `probablyfine/checker.py` — Extend `CHECKER_SYSTEM_PROMPT` with negative examples

---

### 9. Deletion-Ratio False Positive Guard {IMPLEMENTED}
> Implemented: Added deletion-ratio check to should_reflect() in reflection.py. If deletions > 20 lines and exceed additions by 3:1, forces reflection regardless of mode. Placed before FAST mode skip so even fast mode gets safety coverage on large deletions. [2026-05-22]

**Category:** review-quality
**Impact:** 3/5 — No current mechanism detects when edits accidentally delete large amounts of code; cloud agents flag this automatically via their understanding of intent vs. diff.
**Effort:** XS — Add line count heuristic in `reflection.py`, ~15 lines.
**Priority Score:** 15

**The Gap:** Cloud models analyze the semantic relationship between a task and the resulting diff — they understand when "update the footer" shouldn't result in 200 lines deleted. 8B models sometimes produce edits that accidentally delete surrounding code (especially in whole-file fallback mode). The checker doesn't have a specific heuristic for disproportionate deletions.

**Proposed Approach:**
- In `should_reflect()` in `reflection.py`, add a deletion-ratio check: parse the diff to count `+` and `-` lines
- If deletions exceed additions by more than 3:1 AND total deletions > 20 lines, force reflection regardless of mode
- Log the deletion ratio for monitoring
- Consider adding a pre-checker warning displayed to the user: "Warning: diff deletes N lines but only adds M"

**Files Affected:**
- `probablyfine/reflection.py` — Add deletion ratio heuristic to `should_reflect()`

---

### 10. Line-Anchored Edit Format {IMPLEMENTED}
> Implemented: Added `_LINE_ANCHORED_RE` regex and line-range replacement to edit_parser.py. FileEdit model extended with line_start/line_end fields. System prompt in agent.py updated with LINES format description. Supports `FILE: path LINES N-M` followed by `<<<<<<< REPLACE` / `>>>>>>> END` blocks. Graceful fallback on out-of-range lines. [2026-05-22]

**Category:** edit-accuracy
**Impact:** 4/5 — Eliminates the SEARCH block uniqueness requirement that causes many edit failures when 8B models can't produce sufficiently unique search text.
**Effort:** M — New regex in `edit_parser.py`, system prompt update in `agent.py`, ~100 lines across 2 files.
**Priority Score:** 12

**The Gap:** Cloud models with massive context windows can always produce unique SEARCH blocks by including enough surrounding context. 8B models with limited context frequently produce non-unique search text (matching 2+ locations in a file). A line-anchored format like `LINES 42-50:` would bypass this fundamental limitation.

**Proposed Approach:**
- Add a new edit format to `edit_parser.py`: `FILE: path.py LINES 42-50` followed by replacement content
- Add corresponding regex `_LINE_ANCHORED_RE` alongside existing `_SEARCH_REPLACE_RE`
- In `agent.py` system prompt, present line-anchored format as an alternative when the model knows line numbers
- In `_apply_single_edit()`, implement line-range replacement using line numbers from the edit
- Fall back gracefully if line numbers are out of range (file modified since model saw it)

**Files Affected:**
- `probablyfine/edit_parser.py` — Add `_LINE_ANCHORED_RE` regex and line-based apply logic
- `probablyfine/agent.py` — Update system prompt to describe the line-anchored format option

---

### 11. File Size Awareness in Context Budget {IMPLEMENTED}
> Implemented: Added `max_context_bytes` config option (default 48000, ~12k tokens) to config.py. Added `_filter_by_budget()` to file_selector.py that skips files exceeding remaining byte budget with logging. Wired into cli.py callsite via `get_max_context_bytes()`. [2026-05-22]

**Category:** context-management
**Impact:** 3/5 — Large files silently consume the context window, leaving insufficient room for model reasoning; cloud models handle this with 128k+ windows.
**Effort:** S — Add size tracking to `file_selector.py` and `context.py`, ~40 lines across 2 files.
**Priority Score:** 12

**The Gap:** Cloud models have 128k-200k token context windows where file size rarely matters. With 8B models limited to 16k tokens, a single large file can consume the entire context budget, leaving no room for the system prompt, task description, or model reasoning. Neither `file_selector.py` nor `context.py` track or limit by file size.

**Proposed Approach:**
- Add a `size_bytes` property to tracked files in `FileContext` in `context.py`
- In `select_files()` in `file_selector.py`, estimate token count (bytes / 4) and skip files that would exceed remaining context budget
- Add a configurable `max_context_bytes` to `config.py` (default: 48000 = ~12k tokens, leaving 4k for prompt + reasoning)
- Log when files are excluded due to size budget: "Excluding large_file.py (15k tokens) — exceeds remaining budget"

**Files Affected:**
- `probablyfine/context.py` — Add size tracking to `FileContext`
- `probablyfine/file_selector.py` — Add budget-aware file filtering
- `probablyfine/config.py` — Add `max_context_bytes` config option

---

### 12. Model-Specific Prompt Variants {IMPLEMENTED}
> Implemented: Added centralized `_MODEL_PROMPT_SUFFIXES` registry and `get_prompt_suffix()` in ollama_utils.py. Deepseek-coder gets JSON schema reinforcement suffixes for checker, classify, decompose, and file_selector phases. Wired into checker.py (system prompt), interpreter.py (classify + decompose prompts), and file_selector.py (selection prompt). [2026-05-22]

**Category:** prompt-engineering
**Impact:** 4/5 — Checker logs show deepseek-coder:6.7b returning completely non-conforming JSON schemas (`{"status": "success", "data": ...}` instead of `{"verdict": ...}`), proving the same prompt doesn't work across models.
**Effort:** M — Prompt variants for each module that calls LLM, ~100 lines across 3-4 files.
**Priority Score:** 12

**The Gap:** Cloud APIs have consistent instruction-following behavior. Different local models respond very differently to the same prompt — qwen3:8b mostly follows JSON format instructions while deepseek-coder:6.7b frequently invents its own response schema. Using identical prompts for both models wastes the checker entirely when deepseek is active.

**Proposed Approach:**
- Create a prompt registry dict in `ollama_utils.py` or a new `prompts.py`: `{model_name: {phase: prompt_template}}`
- For deepseek-coder, add stronger JSON schema reinforcement — repeat the exact expected keys at the end of the prompt
- For qwen3:8b, continue using `/no_think` suffix where appropriate
- In `checker.py`, `interpreter.py`, and `file_selector.py`, look up model-specific prompt before falling back to default
- Start with checker prompts (highest impact) then expand to others

**Files Affected:**
- `probablyfine/checker.py` — Add model-specific prompt selection
- `probablyfine/interpreter.py` — Add model-specific prompt selection
- `probablyfine/file_selector.py` — Add model-specific prompt selection
- `probablyfine/ollama_utils.py` — Optional: add prompt registry

---

### 13. Replan-on-Failure with Context Update {IMPLEMENTED}
> Implemented: Added `_build_replan_prompt()` and `_MAX_REPLANS=1` to agent.py. After the main execution loop, if steps were skipped due to dependency failures, invokes `interpret_task()` with a summary of what succeeded/failed, then executes the new plan via recursive `execute_plan()` with depth cap. [2026-05-22]

**Category:** multi-file-reasoning
**Impact:** 4/5 — When mid-plan steps fail, the agent cannot adapt; cloud models re-evaluate and adjust their approach based on what they've learned so far.
**Effort:** M — Modify `execute_plan()` to re-invoke interpreter on step failure, ~100 lines in `agent.py`.
**Priority Score:** 12

**The Gap:** Cloud agents dynamically adjust their plans when a step fails — they understand what went wrong, update their mental model, and try a different approach. The current agent either continues blindly (cascading failure) or falls back to whole-file replacement. There is no mechanism to "replan" with updated context after partial execution.

**Proposed Approach:**
- In `execute_plan()` in `agent.py`, when a step fails and has unexecuted dependents, trigger a replan
- Build a "replan prompt" that includes: original task, what succeeded, what failed and why, current file state
- Call `interpret_task()` from `interpreter.py` with the replan prompt and updated file context
- Execute the new plan's remaining steps
- Cap replanning to 1 attempt per execution to avoid infinite loops
- Log replan events with before/after step comparison

**Files Affected:**
- `probablyfine/agent.py` — Add replan trigger and re-invoke logic in `execute_plan()`
- `probablyfine/interpreter.py` — May need a lightweight `replan_task()` variant

---

### 14. Dynamic num_predict by Step Type {IMPLEMENTED}
> Implemented: Added `_STEP_NUM_PREDICT` lookup dict and `_get_step_budget()` helper in agent.py. Edit=4096, create=6144, explain=4096, read=512, verify=512, delete=256. Applied to both `_execute_edit` and `_execute_explain` streaming calls. [2026-05-22]

**Category:** token-efficiency
**Impact:** 2/5 — Saves tokens and reduces generation time for lightweight steps, but doesn't directly improve output quality.
**Effort:** XS — Lookup table in `agent.py`, ~10 lines.
**Priority Score:** 10

**The Gap:** Cloud APIs charge per-token but have effectively unlimited generation budgets. With 8B models, oversized `num_predict` on simple steps (explain, verify) wastes time and VRAM, while undersized budgets on complex steps (create) cause truncation. Fixed `num_predict=4096` is a compromise that's wrong for most steps.

**Proposed Approach:**
- Add a step-type → num_predict mapping in `agent.py`: `{"explain": 4096, "edit": 2048, "create": 6144, "read": 512, "verify": 512, "delete": 256}`
- In `execute_step()`, look up the step's action to set `num_predict` dynamically
- Make the mapping configurable via `config.py` under `[agent]` section
- Log the per-step budget for monitoring

**Files Affected:**
- `probablyfine/agent.py` — Add step-type budget lookup in step execution
- `probablyfine/config.py` — Optional: add configurable budget overrides

---

### 15. Context Utilization Tracking {IMPLEMENTED}
> Implemented: Added `log_token_usage()` helper in ollama_utils.py that extracts prompt_eval_count/eval_count from Ollama responses and logs utilization percentage to tokens.log. Wired into interpreter.py (non-streaming classify/validate calls) and file_selector.py. [2026-05-22]

**Category:** token-efficiency
**Impact:** 2/5 — Pure observability improvement; enables data-driven optimization of context budgets but has no direct quality impact.
**Effort:** XS — Add logging after each LLM call in `ollama_utils.py`, ~15 lines.
**Priority Score:** 10

**The Gap:** Cloud providers offer token usage metrics in every API response. Ollama provides `eval_count` and `prompt_eval_count` in responses but probablyfine doesn't track or log them. Without this data, context budget tuning is guesswork.

**Proposed Approach:**
- In `extract_content()` in `ollama_utils.py`, also extract `eval_count` and `prompt_eval_count` from the response
- Add a `log_token_usage(phase, model, prompt_tokens, completion_tokens, num_ctx)` helper
- Call it after every non-streaming LLM call in `interpreter.py`, `file_selector.py`, and `checker.py`
- Log format: `[tokens] phase=classify model=qwen3:8b prompt=1234/16384 (7.5%) completion=456/800`

**Files Affected:**
- `probablyfine/ollama_utils.py` — Add token extraction and logging helper

---

### 16. Edit Count Capping with Multi-Turn Continuation {IMPLEMENTED}
> Implemented: Added `MAX_EDITS_PER_RESPONSE=10` constant in edit_parser.py. In agent.py, `_execute_edit()` caps parsed edits to 10, applies them, then re-invokes the model with a continuation prompt for remaining edits (up to 3 rounds). Added `_CONTINUATION_TEMPLATE` and `_MAX_CONTINUATION_ROUNDS=3`. [2026-05-22]

**Category:** edit-accuracy
**Impact:** 3/5 — Prevents the worst-case edit storms (74 edits in one response) but these are infrequent; capping improves reliability of each individual edit.
**Effort:** M — Changes in `agent.py` (prompt + continuation logic) and `edit_parser.py` (cap), ~100 lines.
**Priority Score:** 9

**The Gap:** Cloud models can reliably produce dozens of coherent edits in a single response because they maintain precise state across their large context windows. 8B models producing 74 SEARCH/REPLACE blocks in one response (as seen in edit_parser.log) inevitably degrade in quality — later edits reference stale state and produce mismatches.

**Proposed Approach:**
- Add `MAX_EDITS_PER_RESPONSE = 10` constant in `edit_parser.py`
- In `parse_edits()`, if more than `MAX_EDITS_PER_RESPONSE` blocks are found, only return the first N
- In `agent.py`, after applying capped edits, if more were parsed, re-invoke the model with "Continue editing — the following edits remain:" prompt
- Add iteration cap (max 3 continuation rounds) to prevent infinite loops
- Log cap events: "Capped 74 edits to 10, triggering continuation"

**Files Affected:**
- `probablyfine/edit_parser.py` — Add `MAX_EDITS_PER_RESPONSE` cap
- `probablyfine/agent.py` — Add multi-turn continuation logic when edits are capped

---

### 17. Cross-File Consistency Check {IMPLEMENTED}
> Implemented: Added `_verify_cross_file_consistency()` in agent.py — uses ast.parse to extract definitions and imports from changed Python files, checks that imported names still exist in their source files. Runs after all steps complete in execute_plan(), reports warnings non-blockingly. [2026-05-22]

**Category:** multi-file-reasoning
**Impact:** 3/5 — Catches cross-file reference breaks (renamed function not updated in importers), a common 8B model error on multi-file tasks.
**Effort:** M — New function in `agent.py` or new module, AST-based import scanning, ~120 lines.
**Priority Score:** 9

**The Gap:** Cloud models can mentally track all cross-file references and ensure consistency across a multi-file edit. 8B models frequently rename a function in one file but forget to update callers in other files, or add an import that references a non-existent module. No post-edit verification currently exists.

**Proposed Approach:**
- Add a `verify_cross_file_consistency(changed_files)` function in `agent.py`
- For Python files: use `ast.parse()` to extract imports and function/class definitions from each changed file
- Check that all imports resolve to existing modules/functions in the project
- Check that renamed identifiers are updated in all files that reference them
- Run after `apply_edits_atomic()` completes successfully
- Report inconsistencies as warnings (don't block, but log for checker to review)

**Files Affected:**
- `probablyfine/agent.py` — Add post-edit consistency verification function

---

### 18. Step Dependency Validation (Rule-Based) {IMPLEMENTED}
> Implemented: Replaced LLM-based `_validate_plan()` with pure-Python topological sort (Kahn's algorithm). Rules: read before edit on same file, create before edit, verify always last. Removed `VALIDATE_TIMEOUT`, `VALIDATE_NUM_PREDICT` constants and `VALIDATE_PROMPT` template. [2026-05-22]

**Category:** task-decomposition
**Impact:** 2/5 — The LLM validator mostly returns "unchanged" and times out 6/8 times; replacing it saves time but doesn't change output quality much.
**Effort:** S — Replace `_validate_plan()` LLM call with topological sort in `interpreter.py`, ~40 lines.
**Priority Score:** 8

**The Gap:** Cloud models can validate plan ordering as part of their reasoning process. The current approach uses a separate LLM call to validate step ordering, which times out 75% of the time (6/8 attempts in logs). A rule-based topological sort would be instant and deterministic.

**Proposed Approach:**
- Replace `_validate_plan()` in `interpreter.py` with a pure-Python topological sort
- Rules: "read" before "edit" for the same file; "create" before "edit" for new files; "verify" always last
- Detect and break circular dependencies by removing the weakest edge
- Remove the `VALIDATE_TIMEOUT` and `VALIDATE_NUM_PREDICT` constants (no longer needed)
- Keep the function signature for backward compatibility

**Files Affected:**
- `probablyfine/interpreter.py` — Replace `_validate_plan()` with rule-based validation

---

## Existing Items Referenced
| Source File | Item # | Title | Status |
|---|---|---|---|
| next-improvements.md | 18 | Cross-File Import Chain Analysis | OPEN |
| next-improvements.md | 11 | Diff-Aware Retry Prompts | OPEN |
| next-improvements.md | 14 | Pre-Edit Syntax Validation | OPEN |
| next-improvements.md | 9 | Schema Validation for LLM Responses | OPEN |

## Priority Matrix
| Score | Feature | Impact | Effort | Category |
|-------|---------|--------|--------|----------|
| 25 | Few-Shot Edit Examples in System Prompt | 5 | XS | prompt-engineering |
| 20 | Zero-Token Detection and Auto-Retry | 5 | S | error-recovery |
| 16 | Indentation-Aware Fuzzy Matching | 4 | S | edit-accuracy |
| 16 | File-Aware Decomposition | 4 | S | task-decomposition |
| 16 | Cascading Failure Prevention | 4 | S | error-recovery |
| 15 | Hallucination Validator Enhancement | 3 | XS | context-management |
| 15 | Decomposition Timeout Resilience | 3 | XS | task-decomposition |
| 15 | Negative Examples for Checker | 3 | XS | prompt-engineering |
| 15 | Deletion-Ratio False Positive Guard | 3 | XS | review-quality |
| 12 | Line-Anchored Edit Format | 4 | M | edit-accuracy |
| 12 | File Size Awareness in Context Budget | 3 | S | context-management |
| 12 | Model-Specific Prompt Variants | 4 | M | prompt-engineering |
| 12 | Replan-on-Failure with Context Update | 4 | M | multi-file-reasoning |
| 10 | Dynamic num_predict by Step Type | 2 | XS | token-efficiency |
| 10 | Context Utilization Tracking | 2 | XS | token-efficiency |
| 9 | Edit Count Capping with Multi-Turn Continuation | 3 | M | edit-accuracy |
| 9 | Cross-File Consistency Check | 3 | M | multi-file-reasoning |
| 8 | Step Dependency Validation (Rule-Based) | 2 | S | task-decomposition |

## Empirical Data
| Log File | Entries Analyzed | Key Metrics |
|---|---|---|
| agent.log | 100 lines, ~15 sessions | Tier 2 retries: 5 (mixed success), Tier 3 fallbacks: 1, zero-token stalls: 1 (171s wasted), edit match rate warnings: 3 (0/2, 0/2, 2/3) |
| checker.log | 307 lines, 22 sessions | PASS: 14 (64%), FAIL: 3 (14%), zero-token timeout: 5 (23%), wrong JSON schema: 3, avg time: 50-100s per check |
| edit_parser.log | 50 lines | Max edits/response: 74, SEARCH not found: 4, fuzzy match success: 1, atomic rollback: 1, new file creates: 3 |
| file_selector.log | 183 lines, ~25 selections | Hallucinated paths: ~30% of selections, README.md hallucinated in 80% of cases, all-paths-hallucinated: 3 events, Ollama connection error: 1 |
| interpreter.log | 708 lines, ~30 sessions | Classification empty: 5, decomposition timeout: 5 (60s), validation timeout: 6 (25s), fallback plans: 8, truncated JSON repairs: 8, clarity < threshold: 6 |
