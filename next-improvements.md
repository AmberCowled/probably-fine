# Next Improvements — probablyfine

Prioritized ideas for the next wave of features, informed by the current codebase (~5,100 lines, 22 modules), session log analysis, and research on agent architectures.

---

## Tier 1 — High Impact, Direct Path

### 1. Replace Aider with Custom Agent (`agent.py`)

The single biggest unlock. Aider is used in narrow single-shot `--message` mode — we don't use its chat, tools, or history features. Replacing it gives us:

- **Full streaming visibility** during the maker phase (currently opaque subprocess)
- **Hang detection during code generation** (only checker has it today)
- **DRM integration** throughout the maker phase (currently blind to VRAM during Aider)
- **Simpler diff capture** — direct before/after file comparison instead of the 68-line 4-fallback-chain in `aider_session.py`
- **SEARCH/REPLACE edit format** with validation (SEARCH block must match file content before applying)

Estimated ~430 lines. All prerequisite patterns exist in `checker.py` (streaming), `ollama_utils.py` (client factory), and `file_selector.py` (structured output parsing). See `report.md` Part 3 for full architecture.

**Open questions:** RepoMap equivalent for AUTO mode, edit format reliability with qwen3:8b, multi-file edit markers.

### 2. Destructive Edit Protection

Three consecutive session runs showed the maker deleting 50-78% of CSS files on simple UI tasks. This is the most user-visible quality problem.

- **Conservative edits mode:** Append "Make minimal, targeted changes. Do not rewrite or delete existing code unless explicitly asked." to task prompts. Config toggle: `[aider] conservative_edits = true`.
- **Diff size warning:** If maker changes >50% of any single file, show a warning before proceeding to checker.
- **Deletion-ratio gate:** If >50% of diff lines are deletions, auto-escalate to FAIL or require manual confirmation regardless of checker verdict.
- **Post-checker confidence gate:** If confidence < 60% AND diff > 100 lines, warn: "Low-confidence PASS on large diff — review manually."

### 3. Living Plans with Re-evaluation

The interpreter produces a `TaskPlan` with ordered steps, but today those steps are executed without checking if earlier steps changed assumptions. Inspired by Devin's dynamic re-planning:

- After each step completes, re-read affected files and compare against expected state
- If a file was modified in an unexpected way (by a previous step), regenerate remaining steps with updated context
- Detect when a step's output contradicts the plan's intent (specification drift from Wink research)

This matters most for complexity-2+ tasks where steps have dependencies.

---

## Tier 2 — Meaningful Quality Improvements

### 4. Misbehavior Observer

Lightweight monitor running during execution that watches for three failure patterns (from Wink research, which found these in ~30% of agent trajectories):

| Pattern | Detection | Recovery |
|---|---|---|
| Specification drift | Output diverges from original task intent | Re-inject original prompt as context |
| Reasoning loops | Same failing action attempted 2+ times | Force alternative approach |
| Tool failures | Malformed edits, hallucinated file paths | Validate against actual codebase, retry with error context |

Could be a simple observer function called between execution steps. Single-intervention recovery succeeds 90% of the time per the research.

### 5. Reflexion on Failure

When a step fails (syntax error, edit doesn't apply, tests fail), prompt the model: "Step N failed with error: {error}. What went wrong and what should we try instead?" Store the reflection as context for the retry. This is the core insight from the Reflexion paper — verbal self-correction is cheap and effective.

Currently the repair loop in `reflection.py` sends checker feedback to Aider, but doesn't ask the model to reason about *why* the failure happened. Adding this intermediate reflection step should improve repair quality.

### 6. Architect/Editor Split via Thinking Mode

Use qwen3:8b's hybrid thinking/fast mode as a lightweight two-phase approach:

- **Architect phase** (thinking mode): Analyze the task, identify files to change, design the approach, produce pseudocode or a high-level diff outline
- **Editor phase** (fast mode, `/no_think`): Given the architect's plan, produce precise SEARCH/REPLACE edits

This mirrors Aider's architect/editor model but uses a single model with mode switching instead of two separate models. No model swap needed — just different system prompts and the thinking toggle.

### 7. Escalation to Planning Model (Reflection Phase 5)

The ESCALATE verdict exists in the checker response format but does nothing today. Implement the full path:

- When checker returns ESCALATE, re-run the review with the planning model (qwen3:8b in thinking mode) and an extended prompt that includes architectural assessment
- Skip escalation if checker_model == planning_model (currently the case — wait until there's a meaningful model tier difference)
- Show escalation clearly in CLI output

Lower priority until the model lineup includes a genuinely stronger review model.

### 8. Session Memory / Conversation Context

probablyfine currently treats every task as independent — no memory of what was done earlier in the session. Adding lightweight session context:

- Track which files were modified and what was changed (summary, not full diffs)
- On subsequent tasks, include a brief "session history" block: "Earlier this session: added login endpoint to auth.py, updated nav template"
- Detect related follow-up tasks ("now add tests for what we just built") and auto-include relevant files
- Persist per-session, not across sessions. Reset on `/clear`.

This is especially valuable for multi-step workflows where the user iterates on a feature.

---

## Tier 3 — Polish and Infrastructure

### 9. Schema Validation for All LLM Output

Every LLM call that expects structured output (interpreter classification, task decomposition, checker verdicts, file selection) should validate against a schema before use. Currently each module has ad-hoc JSON parsing with regex fallbacks.

- Define JSON schemas for each structured output type (could use `jsonschema` or simple dataclass validation)
- Single validation function in `ollama_utils.py`: `validate_llm_response(raw, schema) -> parsed | fallback`
- Reduces per-module parsing boilerplate and catches malformed responses earlier

### 10. Repo Map for AUTO Mode

When the user doesn't explicitly `/add` files, the agent needs to discover relevant files. The current `file_selector.py` asks the LLM to pick files from a directory listing. A lightweight repo map would improve this:

- Use Tree-sitter or AST parsing to build a symbol index (function names, class names, imports)
- Rank files by connectivity (files that import/export to many others are more likely relevant)
- Feed a compressed map to the model alongside the task prompt
- Only needed for AUTO mode — explicit file selection bypasses this entirely

### 11. Task Execution Metrics Dashboard

Extend the existing `/resources` command or add `/stats`:

- Per-task timing breakdown: interpret → file select → maker → checker → total
- Token counts per phase (input/output)
- Model swap frequency and cumulative swap time
- Success/failure/repair rates across the session
- Historical trends persisted to `~/.probablyfine/metrics.json`

Useful for understanding where time is spent and whether improvements are actually helping.

### 12. Test Infrastructure

No tests exist today. Before the agent.py replacement (which is a large refactor), establish baseline testing:

- Unit tests for pure functions: JSON parsing, diff truncation, keyword pattern matching, prompt construction
- Integration test harness: mock Ollama responses, verify end-to-end flow from task → plan → (mocked) execution → checker verdict
- Snapshot tests for prompt templates (catch unintended prompt changes)

Even a small test suite would catch regressions during the Aider → agent.py migration.

### 13. Parallel Step Execution (Sequential Inference)

The interpreter's `TaskPlan` includes dependency information (`depends_on`). Steps without mutual dependencies could theoretically execute in any order. On single-GPU hardware, true parallel inference isn't possible, but:

- Identify independent steps and batch their file reads upfront
- Order independent steps to minimize model context switching
- In the future (multi-GPU or API-based inference), this structure enables real parallelism

### 14. Configurable Checker Personality

The checker prompt is hardcoded in `checker.py`. Different tasks benefit from different review emphases:

- **Security-focused**: Deeper analysis of input handling, injection vectors, auth checks
- **Performance-focused**: Flag unnecessary allocations, O(n^2) patterns, missing caching
- **Correctness-focused** (default): Logic errors, edge cases, spec compliance

Could be auto-selected based on task intent from the interpreter, or manually via `/checker security`.

---

## Tier 4 — Future / Exploratory

### 15. Multi-File Edit Transactions

When the agent edits multiple files in one task, apply all changes atomically:

- Stage all edits in memory, validate all SEARCH blocks match current file state
- Apply all edits together, or roll back entirely if any single edit fails
- Uses git stash/restore as the transaction mechanism

### 16. Fallback Control Flow (ReAcTree Pattern)

For complex tasks, allow the plan to specify fallback strategies:

```json
{
  "type": "fallback",
  "primary": { "action": "edit", "description": "Refactor using new API" },
  "fallback": { "action": "edit", "description": "Wrap old API with adapter" }
}
```

If the primary approach fails (edit doesn't apply, tests fail), automatically try the fallback before asking the user.

### 17. Incremental Context Compression

As session history grows, compress earlier context to stay within model limits:

- Summarize completed steps into single-line descriptions
- Keep full context only for the current and immediately preceding steps
- Use the Claude Code pattern of an append-only message log with automatic compression

### 18. Model Upgrade Path

The architecture should accommodate future model improvements without code changes:

- Abstract model capabilities behind a trait system: `supports_thinking`, `supports_tool_calls`, `structured_output_quality`
- When better models become available for Ollama (e.g., Qwen3-14B, DeepSeek-V3-distilled), just update `config.toml` and capability flags
- DRM already handles the VRAM/swap implications — just need the capability layer

### 19. Web Context Integration

For tasks that reference external libraries or APIs, fetch relevant documentation:

- Detect library/API mentions in the task prompt
- Fetch and summarize relevant docs (README, API reference)
- Include as additional context for the maker phase
- Requires internet access — make it opt-in and cacheable

### 20. Voice/Natural Language Commit Messages

After a task completes, auto-generate a descriptive commit message from the diff and task context:

- Use the fast model to summarize: "What changed and why?"
- Present to user for approval/editing before committing
- Integrate with existing git_utils.py
- Already partially supported by Aider's auto-commit — but with agent.py, we'd own this entirely

---

## Summary by Effort

| Effort | Items |
|---|---|
| Quick wins (1 session) | #2 destructive edit protection, #14 checker personality, #9 schema validation |
| Medium (2-3 sessions) | #5 reflexion on failure, #6 architect/editor split, #8 session memory, #20 commit messages |
| Large (3-5 sessions) | #1 agent.py replacement, #3 living plans, #4 misbehavior observer, #12 test infrastructure |
| Ongoing / future | #10 repo map, #15 transactions, #16 fallback control flow, #17 context compression, #18 model upgrades, #19 web context |

---

## Recommended Sequence

```
#2 Destructive edit protection     ← Immediate quality win, small effort
#1 agent.py replacement            ← Foundation for everything else
#12 Test infrastructure            ← Safety net before more changes
#6 Architect/editor split          ← First major agent intelligence upgrade
#5 Reflexion on failure            ← Better repair quality
#3 Living plans                    ← Dynamic re-planning
#4 Misbehavior observer            ← Reliability at scale
#8 Session memory                  ← Better multi-task workflows
```
