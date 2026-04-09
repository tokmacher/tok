# Benchmark Assets

Reportable catalog runs use pinned local snapshot assets instead of fetching repos live.

- Default snapshot path for a task is `benchmarks/assets/<task-id>/workspace`.
- If a task declares `seed_patch`, it is resolved relative to the task asset directory
  unless it is absolute.
- `repo=tok ref=HEAD` tasks may use the local checkout, but reportable runs require a
  clean worktree.
- Use `--local-debug` only for non-reportable development runs.
- Verify the pack before kickoff or release-candidate runs with
  `uv run python scripts/prepare_benchmark_assets.py --root benchmarks verify`.

The executor will fail reportable runs if a required snapshot asset is missing.
