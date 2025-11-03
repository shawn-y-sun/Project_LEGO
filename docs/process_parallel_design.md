# Process-Based Parallel Search Design

This document expands **Option 2** into a full multi-processing design for the exhaustive search pipeline.  Each conceptual step lists the issues it resolves so we can verify the approach covers all previously observed obstacles (shared caches, logging, replay, etc.).

## 0. Rationale for Choosing Multi-processing

* **Bypasses the Python GIL.** Every worker runs in its own interpreter, so pure-Python sections of `assess_spec` (feature validation, filter orchestration) can execute concurrently instead of serialising under the GIL as they do with threads.
* **Matches the read-only cache strategy.** We already pre-build feature tables before the search loop.  Forked processes can reuse these caches via copy-on-write memory or explicit shared-memory blocks with minimal modification.
* **Predictable scaling on many cores.** Machines with 100+ physical cores benefit because each process can occupy a dedicated CPU without fighting thread scheduling overheads.

## 1. Up-front Feature Materialisation & Freeze

1. Run the full feature-engineering pass on the coordinator (main process).
2. Mark the resulting frames as immutable and export them through a narrow API (e.g., `get_readonly_bundle()` returning plain NumPy arrays / Parquet pointers).

**Issues addressed**
* Eliminates shared-cache race conditions because workers never call mutating feature builders.
* Provides a single canonical dataset, keeping memory overhead bounded irrespective of worker count.

## 2. Build a Shared Transport for Feature Bundles

1. Choose a transport depending on dataset size:
   * Small / medium tables → rely on copy-on-write semantics of `fork` so each worker inherits the frozen frames without duplicating pages.
   * Large panels → back frames with memory-mapped files or `multiprocessing.shared_memory` blocks and hand workers the offsets.
2. Package lightweight metadata (column order, dtypes, scenario mapping) alongside the transport handles.

**Issues addressed**
* Avoids deep-cloning `DataManager` per worker while still isolating writes—processes get private views of the same underlying pages.
* Keeps inter-process serialisation costs low because only metadata crosses the pipe.

## 3. Worker Bootstrap & Local Runtime Context

1. Each worker process, upon start, constructs a minimal `SearchContext` containing:
   * Read-only accessors that wrap the shared transport (no mutators exposed).
   * Stateless helpers to instantiate `CM` objects with the frozen data.
2. Any candidate-specific temporary transformations are performed on local copies created inside the worker so they never touch shared buffers.

**Issues addressed**
* Ensures `CM.build` and filter routines operate without needing locks or coordination.
* Prevents accidental mutation of global caches because workers only receive immutable handles.

## 4. Task Submission & Execution

1. The coordinator enumerates specs and submits jobs via `ProcessPoolExecutor` / `multiprocessing.Pool` with the snapshot metadata.
2. Workers execute `assess_spec` independently, timing their phases (optional) and capturing structured outcomes: pass/fail info, errors, plus any feature mutations they performed locally but that might matter to the canonical manager.

**Issues addressed**
* Achieves true parallel evaluation across CPU cores, avoiding the GIL contention that slowed the threaded prototype.
* Maintains deterministic diagnostics because each worker tracks its own timing/log data.

## 5. Result Aggregation & Progress Reporting

1. The coordinator retains ownership of progress UI (tqdm) and shared result containers.
2. As futures complete, the coordinator merges results, de-duplicates passed models, and logs status messages in submission order.
3. Optional: maintain a concurrency summary (start/end timestamps, CPU utilisation) to confirm cores are saturated.

**Issues addressed**
* Prevents stdout interleaving by centralising logging.
* Provides the same deterministic ordering semantics as the legacy serial loop.

## 6. Replay Required Feature Mutations into the Live Manager

1. Workers record any feature constructions that matter for downstream persistence (e.g., transforms not pre-built during the global freeze).
2. After all tasks finish, the coordinator replays the unique mutations on the live `DataManager` while it is unlocked for mutation.
3. Validate that all passed CMs’ requirements exist in the refreshed manager before returning results.

**Issues addressed**
* Keeps the authoritative `DataManager` synchronised with successful models without requiring workers to mutate shared state.
* Guards against drift between the immutable worker view and the mutable canonical cache.

## 7. Resource Management & Safety Nets

1. Expose parameters such as `parallel`, `cores`, and `prefetch_strategy` so segments can tune concurrency.
2. Implement graceful shutdown (context managers) to ensure processes terminate even if a worker raises.
3. Add integrity checks: confirm that shared-memory handles are released, and fallback to serial mode if initialisation fails.

**Issues addressed**
* Protects against zombie processes / leaked shared-memory regions.
* Gives operators knobs to balance CPU usage against memory constraints.

## 8. Optional Enhancements

* **Warm worker pools** that keep processes alive across segments to amortise startup cost.
* **Hybrid scheduling** where pure-Python pre-filtering runs serially, and only heavy model fits are dispatched to the pool.
* **Telemetry hooks** that record per-process CPU usage to verify the multicore advantage in production.

These steps provide a complete, process-based parallel architecture that resolves the pain points encountered with the current threaded implementation while aligning with the read-only data-manager strategy already in place.

## Branch Hosting

The process-based prototype described above lives on the dedicated `process-parallel-option` branch so it can evolve independently from the threading-focused branch. Keeping the two approaches on separate branches makes it easy to benchmark and iterate on each option without intermingling their experiments.
