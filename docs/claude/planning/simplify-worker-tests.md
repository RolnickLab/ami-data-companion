# Simplify Worker Tests

**Date:** 2026-01-28
**Status:** Planned (implement after pipereg PR merges)
**File:** `trapdata/api/tests/test_worker.py`

## Goal

Remove redundant tests that duplicate concepts or test server behavior rather than client behavior. Keep tests focused on validating our client code works correctly.

## Tests to Remove

### 1. Duplicate "empty queue" tests (keep one)

| Test | Line | Action |
|------|------|--------|
| `TestRESTDatasetIntegration.test_empty_queue` | 215 | **KEEP** - tests iterator stops |
| `TestGetJobsIntegration.test_empty_queue` | 278 | REMOVE - same concept |
| `TestProcessJobIntegration.test_empty_queue` | 331 | REMOVE - same concept |

### 2. Duplicate "multiple batches" tests (keep one)

| Test | Line | Action |
|------|------|--------|
| `TestRESTDatasetIntegration.test_multiple_batches` | 225 | REMOVE - covered by E2E |
| `TestWorkerEndToEnd.test_multiple_batches_processed` | 554 | REMOVE - similar to full workflow |

### 3. Error handling variations (keep mixed, remove pure failure)

| Test | Line | Action |
|------|------|--------|
| `TestProcessJobIntegration.test_handles_failed_items` | 384 | REMOVE - pure failure case less realistic |
| `TestProcessJobIntegration.test_mixed_batch_success_and_failures` | 404 | **KEEP** - realistic scenario |

### 4. Implementation details

| Test | Line | Action |
|------|------|--------|
| `TestGetJobsIntegration.test_query_params_sent` | 285 | REMOVE - tests implementation not behavior |

## Tests to Keep

### TestRestCollateFn (unit tests - keep all)
- `test_all_successful` - happy path
- `test_all_failed` - error path
- `test_mixed` - realistic scenario
- `test_single_item` - edge case

These are unit tests of our collation logic, not integration tests.

### TestRESTDatasetIntegration
- `test_fetches_and_loads_images` - core functionality
- `test_image_failure` - error handling for bad URLs
- `test_empty_queue` - iterator termination

### TestGetJobsIntegration
- `test_returns_job_ids` - core functionality

### TestProcessJobIntegration
- `test_processes_batch_with_real_inference` - core ML path
- `test_mixed_batch_success_and_failures` - realistic error scenario

### TestWorkerEndToEnd
- `test_full_workflow_with_real_inference` - complete workflow

### TestRegistrationIntegration
- `test_get_user_projects` - fetch projects
- `test_register_pipelines_for_project` - register pipelines

## Summary

| Before | After | Removed |
|--------|-------|---------|
| 19 tests | 13 tests | 6 tests |

## Implementation

```bash
# After pipereg PR merges, delete these test methods:
# - TestRESTDatasetIntegration.test_multiple_batches
# - TestGetJobsIntegration.test_empty_queue
# - TestGetJobsIntegration.test_query_params_sent
# - TestProcessJobIntegration.test_empty_queue
# - TestProcessJobIntegration.test_handles_failed_items
# - TestWorkerEndToEnd.test_multiple_batches_processed

# Then run tests to verify nothing broke:
pytest trapdata/api/tests/test_worker.py -v
```
