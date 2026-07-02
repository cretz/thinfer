# Agent notes

## Rules

- Keep markdowns in this dir succinct. Bullets over prose. No overviews of what a file is. No restating context.
- Update `worklog.md` as work progresses. Update `plan-details.md` when a design decision lands. Don't duplicate content between files.
- On session start, read `agent-notes.md` (this file), `orig-plan.md`, `plan-details.md`, `worklog.md` (in that order).
- Never use em dashes anywhere (chat, code, comments, docs, commit messages). Use hyphens, colons, parens, or split sentences.
- After TS edits in `thinfer-web`: run `pnpm fmt && pnpm lint && pnpm typecheck`. After Rust edits: `cargo fmt && cargo clippy`. Always all three (fmt included), not just typecheck/clippy.
- Never put weight/activation bytes in WASM linear memory. Use JS heap on web; mmap or `Vec` on native.
- Performance is a first-class concern, including first-time load. Don't add abstractions that force serial work where parallel is possible.
- Worklog is forward-looking. No "done this session" / changelog sections - git history is the record. Keep `In progress`, `Next`, `Carry-forward context` only.
- Code quality needs to be high and succinct and legible.
- Session-start re-familiarization: read the four working-area files and skim the immediately-relevant code, then start writing. Do not survey the whole crate graph or re-read ops you've already seen in prior sessions. If the worklog says "next: build X in file Y", open Y and start. Hard budget for getting up to speed: ~50k tokens. If you're past that without having edited anything, you're over-investigating - start writing.
- Surface the plan before non-trivial edits. State what you're about to do in 2-3 lines and ask if it lands well, especially when adding new ops, files, or crate-level wiring. Don't silently architect.
- Use existing out-of-repo scratch/ dir for logs and PNG dirs and such.

## File guide

- `worklog.md` - START HERE. Forward-looking state + what's next. Points to the active per-model plan. Keep it slim.
- `orig-plan.md` - user's source-of-truth plan. Don't edit.
- `plan-details.md` - engine-wide design + current kernel/runtime/web/operational reference (the "how" beyond orig-plan, reused across models). Per-model porting details go in their own file, not here.
- `hy15-causal-plan.md` - ACTIVE per-model port (HunyuanVideo 1.5 causal I2V, minWM dmd). START HERE for current work.
- `ltx-plan.md` - shipped per-model port (LTX-2.3 distilled-1.1; 22B joint audio-video).
- `wan-plan.md`, `qwen-image-plan.md`, `zimage-plan.md` - shipped/archived per-model references. Read only if touching that model.
- `agent-notes.md` - this file: working conventions.
