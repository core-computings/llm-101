# Repository instructions for LLM 101

## Incremental memory

- Read [incremental_memory.md](incremental_memory.md) before starting work and apply relevant lessons to the current task.
- Every time the user corrects generated work or requests a revision, update `incremental_memory.md` as part of that revision, before finishing the response. This applies to all generated work, including prose, code, diagrams, and repository instructions.
- Summarize the feedback, the concrete change made, and the actionable rule that will prevent the same mistake or unwanted choice from recurring. Record the date and relevant files or scope.
- Update an existing entry when the feedback refines the same lesson. Add a new entry for a distinct lesson, preserve useful history, and avoid duplicate or speculative lessons.
- Distinguish a scoped preference or changed requirement from an actual error. Do not turn a one-off request into a repository-wide rule without evidence.

## SVG diagrams

- Before creating, extending, or repairing instructional SVGs under `docs/**/assets/**/*.svg`, read and follow [svg_specs.md](svg_specs.md).
- `svg_specs.md` contains the complete visual system and mandatory SVG preflight checklist. Keep detailed SVG requirements there rather than duplicating them in this file.
