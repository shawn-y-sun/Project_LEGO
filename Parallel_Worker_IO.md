# Parallel Search Worker Inputs and Outputs

This note details the contract for a process-based worker that evaluates a single specification in the parallel search design.

## Inputs passed into a worker
- **Spec payload**: the concrete combination to evaluate (what the current code calls `spec_combo`), including the `index` assigned by the coordinator for ordering/resume semantics.
- **Serialized configuration**: immutable inputs needed to build the `CM`/`DataManager` in the worker:
  - Dataset configuration (paths, split settings, sample sizes). Every worker **reconstructs its own `DataManager`** from this
    config inside its process; there is no shared in-memory manager. Data access is therefore read-only and independent—no lock
    or queue is required as long as the backing files are concurrently readable. When a run is attached to a `Segment`, pass a
    serialization of the canonical `segment.dm` so the worker clone mirrors the same variable definitions and split logic.
  - Model hyperparameters/options for the current spec.
  - Test configuration and any toggles used by `filter_test_info`.
- **Run-scoped constants**: values that should be identical for every worker invocation:
  - The `start_index` offset so workers can report their absolute position in the global list.
  - The random seed (if deterministic behavior is required across processes).
  - Flags that change behavior (e.g., `keep_cms`, `save_artifacts`).
- **Worker ID hints**: IDs are issued in the coordinator, so the worker receives a `temp_id` token for logging; persistent IDs (e.g., `passed_k`) are finalized after results return to the coordinator.
- **Test-update hook identifier**: a reference (by name or enum) telling the worker which `test_update_func` behavior to use. The function itself should be importable inside the worker process to avoid pickling issues.

## Outputs returned to the coordinator
Workers always return a structured record so the coordinator can serialize progress updates and persistence logic without race conditions.

- **Success result**:
  - `status`: `"passed"`.
  - `spec_index`: the absolute index (start_index + local position) for progress tracking.
  - `temp_id`: echoes the input token so the coordinator can map futures to submitted tasks.
  - `cm_artifact`: the fitted `CM` (or a serialized lightweight form) ready to be saved by the coordinator. If the worker generated new derived
    features or transformed variable names, include a **variable-registry delta** so the coordinator can replay those additions onto the canonical
    segment-level `DataManager` before saving the CM.
  - `filter_test_info`: structured data about which tests ran/observations (used to batch-print new test descriptions in the coordinator).
  - `metrics`: any scalar metrics computed in `assess_spec` that the coordinator needs for ranking.

- **Filtered-out result** (spec fails filters but no exception):
  - `status`: `"filtered"`.
  - `spec_index` and `temp_id` as above.
  - `reason`: short code/string for why the spec failed.
  - `filter_test_info`: to keep the aggregated view consistent.

- **Error result** (exception during assessment):
  - `status`: `"error"`.
  - `spec_index` and `temp_id` as above.
  - `error_type`: class name or error code.
  - `message`: formatted traceback/message for logging and error report output.

## How the contract addresses prior caveats
- **I/O safety**: Workers never write to disk; they only return data. The coordinator owns all persistence and ID assignment, preventing races on progress files and artifact directories.
- **Resume determinism**: By returning `spec_index` and reusing coordinator-issued `temp_id`, results can be reordered safely if futures complete out of order while still logging in the original sequence.
- **Pickling constraints**: Inputs are simple data structures; workers import their own module code and rebuild objects locally, avoiding non-picklable shared state.
- **Progress accounting**: Every result includes `spec_index`, letting the coordinator increment completed counts as futures finish, preserving the existing progress reporting semantics.
- **No shared `DataManager` bottleneck**: Because each worker instantiates its own `DataManager` from serialized config, there is no contention on a shared object or lock. The only shared resource is the underlying dataset storage, which remains read-only across workers. Reattaching each returned CM to the canonical segment-level `DataManager` preserves a single source of truth for variable constructs even though workers used cloned managers.
