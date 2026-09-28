# Engineering rules

Build correct, readable software with the least complexity needed for the requested behavior. Optimize for the next maintainer, not for minimum line count or maximum abstraction.

## Scope and context

- Apply these defaults alongside the current task and the host's instruction hierarchy. Follow more specific, applicable repository conventions; surface conflicts that affect correctness or scope.
- Inspect the relevant entry points, implementation, callers, and tests before editing. Search for existing behavior and reusable components before creating a new one. Read only documentation relevant to the change.
- If present in the repository, read `docs/engineering/REPOSITORY.md` for boundaries and commands. For Python changes, also read `docs/engineering/python.md`; for model/data/evaluation changes, read `docs/engineering/ml.md`. These profiles are optional and are not loaded automatically by their filenames.
- Preserve the user's work and the repository's established package manager, framework, layout, and formatting. Do not introduce a replacement stack to complete a small task.

## Design and implementation

- Implement the complete requested behavior using the smallest coherent design. Avoid speculative features, configurability, plugin systems, or compatibility layers without a current requirement.
- Reuse a sound existing pattern. Introduce an interface, helper, class, or layer only when it expresses a real concept, isolates a meaningful boundary, or removes duplication of the same knowledge.
- Keep each behavior or business rule in one authoritative place. Similar-looking code with different responsibilities does not automatically need a shared abstraction.
- Keep data flow and dependencies explicit. Separate computation, orchestration, and external I/O when that makes their responsibilities clearer; do not create layers merely to match an architecture diagram.
- Prefer cohesive modules, precise names, and straightforward control flow. Avoid boolean mode switches and generic catch-all modules when they hide different responsibilities.
- Extract a function when its name captures a meaningful operation or it improves reuse, isolation, or readability. Avoid chains of wrappers that only forward arguments. A single-use helper is acceptable when it clarifies a complex operation.
- Treat long files, deep nesting, many parameters, and repeated changes across modules as review signals. Split by responsibility, not by an arbitrary line limit. Do not compress readable code into clever one-liners.
- Keep interfaces and data structures as small as their current contract permits. Prefer composition; use inheritance when the domain or framework has a real substitutability requirement.
- Use mature existing libraries for nontrivial standard problems when appropriate. Justify new dependencies by their benefit and maintenance cost; do not recreate a reliable library to reduce the dependency count.

## Changes and removal

- Fix the cause of a problem, not just the observed symptom. Keep unrelated cleanup outside the requested change.
- When replacing behavior, find its callers and integrations, migrate affected references, and remove the superseded implementation within scope. Avoid leaving parallel `new`, `v2`, or `final` implementations without an intentional migration contract.
- Remove imports, configuration, dependencies, fixtures, and documentation made obsolete by the change. Preserve tests that protect supported behavior; replace a test only when its contract has actually changed.
- Confirm a symbol is unused before deleting it. Check exports, public APIs, registration, reflection, framework callbacks, command entry points, and external consumers. No local caller, low coverage, or a static-analysis warning alone is not proof of dead code.
- Do not add commented-out implementations, temporary debug code, empty placeholders, or fallbacks that conceal an unfinished requirement. Keep legitimate incomplete work explicitly identified with its reason and next action.
- Preserve public APIs, persisted data, and supported behavior unless the task authorizes changing them. When a breaking change is required, update the affected consumers and explain migration implications.
- Never overwrite generated or vendored files as a shortcut when their source or generator is the proper place to fix the issue.

## Readability and documentation

- Use names that reveal domain meaning, units, and important invariants. Match the repository's terminology consistently.
- Let code express obvious mechanics. Comments should explain intent, constraints, non-obvious algorithms, tradeoffs, or reasons an apparent simplification is incorrect.
- Document public or non-obvious contracts: inputs, outputs, errors, side effects, units, and invariants where needed. Avoid boilerplate docstrings that simply restate a name or signature.
- Update nearby comments, examples, and docs when the change makes them inaccurate. Prefer the repository's documentation language; otherwise write code comments and docstrings in concise English.

## Correctness and resource use

- Validate untrusted or external data at boundaries. Establish internal invariants instead of repeatedly normalizing or validating the same data throughout the pipeline.
- Handle failures deliberately. Catch errors where recovery or useful translation is possible; preserve the cause. Do not silently swallow exceptions or return plausible success-shaped defaults after failure.
- Own and release resources clearly: files, connections, tasks, processes, locks, and device memory. For relevant I/O paths, define timeout, cancellation, retry bounds, and duplicate-operation behavior.
- Do not add retries, caches, concurrency, or asynchronous code without a concrete need. Consider invalidation, ordering, race conditions, and failure behavior when they are needed.
- Avoid obvious waste: repeated expensive initialization, unnecessary I/O, unbounded accumulation, redundant transformations, and inappropriate algorithmic complexity. Measure representative workloads before claiming a performance improvement.
- Keep credentials and sensitive payloads out of source, fixtures, and logs. Use the repository's established configuration and credential mechanisms.

## Verification and review

- Identify acceptance criteria from the task. For a bug, reproduce the relevant failure where feasible and add a regression test that would fail without the fix.
- Test observable behavior and meaningful failure cases, using expectations independent of the implementation. Mock external boundaries when useful; retain integration coverage for the contracts that mocks cannot prove.
- Match verification to the change. Documentation-only or formatting-only edits do not need artificial behavior tests. Model-quality and performance claims require appropriate evaluations, not just unit tests.
- Use focused checks during iteration, then the applicable repository quality gate before handoff. Reuse valid results for unchanged code; do not repeatedly run expensive suites without a reason.
- Do not lower thresholds, disable checks, add blanket suppressions, weaken assertions, or alter rules merely to make a failing change appear valid. A legitimate exception must be narrow, explained, and consistent with the intended contract.
- Review the final diff for incomplete requirements, regressions, duplication, dead code, excessive indirection, stale comments, unintended dependencies, and changes outside scope.
- Report what changed, the checks actually run and their results, and material limitations. Distinguish existing failures and unavailable checks from failures introduced by the change. Never claim verification you did not perform.

## Code review rules

- Flag concrete correctness, maintainability, or requirement violations with a location and consequence. Distinguish blocking defects from optional improvements.
- Evaluate whether new abstractions reduce complexity for current callers and whether superseded paths were removed safely. Passing checks do not by themselves prove good architecture.
- Keep feedback focused on the changed behavior and its affected contracts. Avoid requesting speculative generalization, broad cleanup, or tests that merely mirror the implementation.