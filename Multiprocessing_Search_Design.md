# Multiprocessing Design for Exhaustive Search

This document describes a process-based architecture to parallelize the model/spec search across CPU cores without altering the user-facing behavior of `run_search`.

## Goals
- Exploit multiple cores for the CPU-bound `assess_spec` work.
- Preserve deterministic IDs, progress/resume semantics, and existing log/UX output.
- Avoid races on artifacts and progress files by centralizing disk writes.
- Keep worker inputs picklable and self-contained.

## High-level flow
1. **Coordinator startup**
   - Build `all_specs` as today and compute `start_index` (for resume) and `total_count`.
   - Create a `ProcessPoolExecutor` (or `multiprocessing.Pool`) with `max_workers` derived from user input or CPU count.

2. **Task submission (bounded in-flight queue)**
   - Issue `temp_{k}` IDs in order and submit work items until an in-flight cap is reached (e.g., `max_workers * 2`).
   - Each submitted task receives only serializable payloads (spec, dataset/model config, `temp_id`, `start_index`, flags, seed).

3. **Worker execution (per-process)**
   - Rebuild its own `DataManager`/model objects from serialized configs—no shared managers, no locks.
   - Run `assess_spec`, produce `filter_test_info`, metrics, and either a fitted `CM` artifact or a filtered/error record.
   - Never writes to disk; returns structured result.

4. **Result handling in coordinator**
   - Consume futures as they complete; reorder by `spec_index` to keep logs/progress deterministic.
   - Persist passed CMs, update `passed_*` indexes, append `failed_info`/`error_log`, and emit progress callbacks/logs.
   - Continue submitting more specs as capacity frees up until all are dispatched.

5. **Shutdown and ranking**
   - Close the pool, then perform the existing ranking/scoring steps on the collected results.

## Addressing prior caveats
- **GIL and core usage**: Processes bypass the GIL, so CPU-bound fitting/testing scales with core count.
- **I/O safety**: Only the coordinator writes to `passed_cms_dir`, progress files, and logs; workers are read-only, preventing races.
- **Resume determinism**: Coordinator-owned `temp_id`/`spec_index` pairs travel with each task; completed results are applied in order so `passed_k` numbering and progress files stay monotonic.
- **Pickling**: Inputs are plain data (spec definitions, primitives, file paths). Workers import modules locally and construct needed objects inside the process, avoiding shared mutable state.
- **Backpressure/memory**: Bounded submission keeps at most O(workers) results in memory and prevents `all_specs` from overwhelming RAM.
- **Test-info aggregation**: Workers return `filter_test_info`; coordinator merges and batch-prints as in the serial flow for a consistent console experience.

## Configuration knobs
- `max_workers`: default to `os.cpu_count()`; allow user override.
- `in_flight_limit`: optional multiplier to tune throughput vs. memory.
- `serialization_mode`: choose between returning full CM objects (for small models) or lightweight serialized artifacts to reduce inter-process transfer.
- `seed_strategy`: fixed seed per worker for reproducibility or derived from `temp_id` for variability.

## DataManager and Segment alignment
- Treat the **segment-attached DataManager as the canonical owner** of variable constructs. Before launching workers, serialize the configuration of `segment.dm` (loaders, split boundaries, transforms) and pass that payload into every task so each worker builds an isolated clone with identical settings.
- When a worker returns a passed CM, the coordinator should **reattach the canonical segment DataManager to the CM before persisting** (e.g., `cm.dm = segment.dm`). This keeps the CM’s variable registry in sync with the segment-level manager and lets downstream `Segment.load_cms` observe the same driver/independent-variable definitions.
- If workers emit additional derived constructs (e.g., auto-generated transforms), include them in the returned payload so the coordinator can **merge those specs into the segment-level DataManager**—either by replaying the feature-building calls on `segment.dm` or by merging a serialized variable registry before saving artifacts.
- Because all persistence still happens in the coordinator, the segment’s `cms/<search_id>/passed_cms` tree stays consistent while preserving the single source of truth for dataset setup.

## Migration plan (conceptual)
- Introduce a pool-backed `filter_specs` path guarded by a `use_multiprocessing` flag while preserving the current threaded/serial codepath for compatibility.
- Wrap the new logic behind a feature flag and add integration tests later once the process pool can be exercised in CI.
