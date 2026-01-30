# Pipeline Registration Branch (pipereg) - Improvement Plan

**Date:** 2026-01-28
**Branch:** `carlos/pipereg` (now up-to-date with `origin/carlosg/pulldl` including PR #104)

## Current State Summary

### What pulldl + PR #104 Added

1. **`get_http_session()` utility** (`trapdata/api/utils.py:41-90`)
   - Creates `requests.Session` with persistent auth header
   - Uses `urllib3.Retry` with exponential backoff (0.5s, 1s, 2s)
   - Uses `HTTPAdapter` for connection pooling
   - Only retries 5XX errors (not 4XX client errors)
   - Context manager for automatic cleanup

2. **Pydantic schemas** for Antenna API contract (`trapdata/api/schemas.py:285-394`)
   - `AntennaPipelineProcessingTask` - task from queue
   - `AntennaJobsListResponse` / `AntennaTasksListResponse` - API responses
   - `AntennaTaskResult` / `AntennaTaskResultError` - results posted back
   - `AsyncPipelineRegistrationRequest` - pipeline registration

3. **Worker functions using sessions** (`trapdata/cli/worker.py`)
   - `post_batch_results()` - uses `get_http_session()` with context manager
   - `_get_jobs()` - uses `get_http_session()` with context manager
   - `_process_job()` - passes `Settings` to above functions

4. **RESTDataset with persistent sessions** (`trapdata/api/datasets.py`)
   - `api_session` for Antenna API calls (with auth)
   - `image_fetch_session` for image downloads (without auth - security)
   - `__del__` method for session cleanup

5. **Integration tests** (`trapdata/api/tests/test_worker.py`)
   - `TestRestCollateFn` - unit tests for batch collation
   - `TestRESTDatasetIntegration` - real image loading
   - `TestGetJobsIntegration` - job fetching
   - `TestProcessJobIntegration` - full ML pipeline
   - `TestWorkerEndToEnd` - complete workflow

6. **Mock Antenna API server** (`trapdata/api/tests/antenna_api_server.py`)
   - `/api/v2/jobs` - list jobs
   - `/api/v2/jobs/{id}/tasks` - get tasks (atomic dequeue)
   - `/api/v2/jobs/{id}/result/` - post results

7. **Test utilities** (`trapdata/api/tests/utils.py`)
   - `patch_antenna_api_requests()` - patches `Session.get/post` for TestClient

### What pipereg Adds (Unique to this Branch)

1. **Pipeline registration** (`trapdata/cli/worker.py:310-500`)
   - `get_user_projects()` - fetch accessible projects
   - `register_pipelines_for_project()` - register for single project
   - `register_pipelines()` - orchestrate registration for multiple projects

2. **CLI command** (`trapdata/cli/base.py:131-164`)
   - `ami register --name "Service Name" --project 1 --project 2`

---

## Issues Identified

### 1. Registration Functions Don't Use `get_http_session()` (High Priority)

**Current state:** `get_user_projects()` and `register_pipelines_for_project()` use raw `requests.get/post` with manual header management, inconsistent with the rest of the codebase.

**Locations:**
- `trapdata/cli/worker.py:327` - `requests.get()` in `get_user_projects()`
- `trapdata/cli/worker.py:376` - `requests.post()` in `register_pipelines_for_project()`

**Problems:**
- No retry logic for transient failures
- No connection pooling
- Inconsistent with worker functions that use `get_http_session()`
- Manual header management duplicated

**Recommendation:** Refactor to use `get_http_session()`:
```python
def get_user_projects(base_url: str, auth_token: str) -> list[dict]:
    with get_http_session(auth_token=auth_token) as session:
        url = f"{base_url.rstrip('/')}/api/v2/projects/"
        response = session.get(url, timeout=30)
        # ...
```

### 2. URL Path Inconsistency (Medium Priority)

**Current state:** Registration functions include `/api/v2` in their URLs:
- `f"{base_url.rstrip('/')}/api/v2/projects/"` (registration)
- `f"{base_url.rstrip('/')}/jobs"` (worker)

The `antenna_api_base_url` setting should either:
- Always include `/api/v2` (and registration functions shouldn't add it)
- Never include `/api/v2` (and all functions should add it)

**Recommendation:** Standardize on `base_url` including `/api/v2` and update registration functions to match worker pattern.

### 3. Missing Tests for Registration (Medium Priority)

**Current state:** No tests for `register_pipelines()`, `get_user_projects()`, or `register_pipelines_for_project()`.

**Recommendation:** Add to `test_worker.py`:
- Add mock endpoints to `antenna_api_server.py`:
  - `GET /api/v2/projects/`
  - `POST /api/v2/projects/{id}/pipelines/`
- Add `TestRegisterPipelinesIntegration` class

### 4. Environment Variable Naming (Low Priority)

**Current state:** Registration uses `ANTENNA_API_TOKEN` while settings use `AMI_ANTENNA_API_AUTH_TOKEN`.

**Recommendation:** Use settings pattern consistently:
```python
settings = read_settings()
auth_token = settings.antenna_api_auth_token
```

---

## Implementation Plan

### Phase 1 & 2: Refactor Registration Functions

**Goal:** Update `get_user_projects()` and `register_pipelines_for_project()` to:
1. Use `get_http_session()` instead of raw `requests.get/post`
2. Use URL pattern consistent with worker functions (base_url already includes `/api/v2`)

**Files to modify:** `trapdata/cli/worker.py`

---

#### Change 1: Update `get_user_projects()` (lines 310-341)

**BEFORE:**
```python
def get_user_projects(base_url: str, auth_token: str) -> list[dict]:
    try:
        url = f"{base_url.rstrip('/')}/api/v2/projects/"
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Token {auth_token}"

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        # ...
```

**AFTER:**
```python
def get_user_projects(
    base_url: str,
    auth_token: str,
    retry_max: int = 3,
    retry_backoff: float = 0.5,
) -> list[dict]:
    """
    Fetch all projects the user has access to.

    Args:
        base_url: Base URL for the API (should NOT include /api/v2)
        auth_token: API authentication token
        retry_max: Maximum retry attempts for failed requests
        retry_backoff: Exponential backoff factor in seconds

    Returns:
        List of project dictionaries with 'id' and 'name' fields
    """
    with get_http_session(
        auth_token=auth_token,
        max_retries=retry_max,
        backoff_factor=retry_backoff,
    ) as session:
        try:
            url = f"{base_url.rstrip('/')}/projects/"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            projects = data.get("results", [])
            if isinstance(projects, list):
                return projects
            else:
                logger.warning(f"Unexpected projects format from {url}: {type(projects)}")
                return []
        except requests.RequestException as e:
            logger.error(f"Failed to fetch projects from {base_url}: {e}")
            return []
```

**Key changes:**
- Add `retry_max` and `retry_backoff` parameters with defaults
- Use `get_http_session()` context manager
- Remove manual `headers` dict (session handles auth)
- Remove `/api/v2` from URL (base_url should include it, matching `_get_jobs` pattern)

---

#### Change 2: Update `register_pipelines_for_project()` (lines 344-401)

**BEFORE:**
```python
def register_pipelines_for_project(
    base_url: str,
    auth_token: str,
    project_id: int,
    service_name: str,
    pipeline_configs: list,
) -> tuple[bool, str]:
    try:
        registration_request = AsyncPipelineRegistrationRequest(...)

        url = f"{base_url.rstrip('/')}/api/v2/projects/{project_id}/pipelines/"
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Token {auth_token}"

        response = requests.post(url, json=..., headers=headers, timeout=60)
        # ...
```

**AFTER:**
```python
def register_pipelines_for_project(
    base_url: str,
    auth_token: str,
    project_id: int,
    service_name: str,
    pipeline_configs: list,
    retry_max: int = 3,
    retry_backoff: float = 0.5,
) -> tuple[bool, str]:
    """
    Register all available pipelines for a specific project.

    Args:
        base_url: Base URL for the API (should NOT include /api/v2)
        auth_token: API authentication token
        project_id: Project ID to register pipelines for
        service_name: Name of the processing service
        pipeline_configs: Pre-built pipeline configuration objects
        retry_max: Maximum retry attempts for failed requests
        retry_backoff: Exponential backoff factor in seconds

    Returns:
        Tuple of (success: bool, message: str)
    """
    with get_http_session(
        auth_token=auth_token,
        max_retries=retry_max,
        backoff_factor=retry_backoff,
    ) as session:
        try:
            registration_request = AsyncPipelineRegistrationRequest(
                processing_service_name=service_name, pipelines=pipeline_configs
            )

            url = f"{base_url.rstrip('/')}/projects/{project_id}/pipelines/"
            response = session.post(
                url,
                json=registration_request.model_dump(mode="json"),
                timeout=60,
            )
            response.raise_for_status()

            result_data = response.json()
            created_pipelines = result_data.get("pipelines_created", [])
            return True, f"Created {len(created_pipelines)} new pipelines"

        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 400:
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("detail", str(e))
                except Exception:
                    error_detail = str(e)
                return False, f"Registration failed: {error_detail}"
            else:
                return False, f"Network error during registration: {e}"
        except Exception as e:
            return False, f"Unexpected error during registration: {e}"
```

**Key changes:**
- Add `retry_max` and `retry_backoff` parameters with defaults
- Use `get_http_session()` context manager
- Remove manual `headers` dict (session handles auth, Content-Type is automatic for json=)
- Remove `/api/v2` from URL
- Fix error handling: use `hasattr()` check instead of `e.response` which may not exist

---

#### Change 3: Update `register_pipelines()` call sites (lines 448, 476-482)

Update the calls to pass through retry settings or use defaults:

```python
# Line ~448: Update get_user_projects call
all_projects = get_user_projects(base_url, auth_token)
# No change needed - defaults are fine

# Lines ~476-482: Update register_pipelines_for_project call
success, message = register_pipelines_for_project(
    base_url=base_url,
    auth_token=auth_token,
    project_id=project_id,
    service_name=full_service_name,
    pipeline_configs=pipeline_configs,
)
# No change needed - defaults are fine
```

---

#### Change 4: Update URL in `register_pipelines()` default (line 421)

The default base_url should include `/api/v2` to match the worker convention:

**BEFORE:**
```python
if base_url is None:
    base_url = os.environ.get("ANTENNA_API_BASE_URL", "http://localhost:8000")
```

**AFTER:**
```python
if base_url is None:
    base_url = os.environ.get("ANTENNA_API_BASE_URL", "http://localhost:8000/api/v2")
```

---

#### Verification

After making changes, run tests:
```bash
pytest trapdata/api/tests/test_worker.py -v
```

The existing tests should still pass since they don't test registration yet.

### Phase 3: Add Registration Tests
1. Add mock endpoints to `antenna_api_server.py`:
   - `GET /api/v2/projects/` - return list of projects
   - `POST /api/v2/projects/{id}/pipelines/` - accept registration
2. Add `TestRegisterPipelinesIntegration` tests:
   - `test_get_user_projects_returns_list`
   - `test_register_pipelines_for_project_success`
   - `test_register_pipelines_for_project_already_exists`
   - `test_register_pipelines_full_workflow`

**Files:** `trapdata/api/tests/antenna_api_server.py`, `trapdata/api/tests/test_worker.py`

### Phase 4: Use Settings Pattern ✅ DONE (2026-01-28)
1. ✅ Updated `register_pipelines()` to accept `Settings` parameter, calls `read_settings()` if None
2. ✅ Removed direct `os.environ.get()` calls, now uses `settings.antenna_api_*`
3. ✅ Fixed env var name in error message (`AMI_ANTENNA_API_AUTH_TOKEN` not `ANTENNA_API_TOKEN`)
4. ✅ Updated `get_http_session()` to read retry settings from Settings when not explicitly provided
5. ✅ Simplified `get_user_projects()` and `register_pipelines_for_project()` - removed retry params

**Files changed:** `trapdata/cli/worker.py`, `trapdata/api/utils.py`

#### Follow-up: Other callers of `get_http_session()` in base branch
These still pass explicit retry values but could be simplified since `get_http_session()` now reads from settings:

1. **`post_batch_results()`** (`trapdata/cli/worker.py:51-55`) - passes `settings.antenna_api_retry_*` explicitly
2. **`_get_jobs()`** (`trapdata/cli/worker.py:66-91`) - has `retry_max`/`retry_backoff` params with hardcoded defaults (3, 0.5)
3. **`RESTDataset`** (`trapdata/api/datasets.py:155-164`) - passes `self.retry_*` from constructor
4. **`get_rest_dataloader()`** (`trapdata/api/datasets.py:364-370`) - passes settings retry values to RESTDataset

These work correctly but are verbose. Consider simplifying to just `get_http_session(auth_token=token)` and letting it read settings internally.

---

## Files Summary

| File | Status | Changes Needed |
|------|--------|----------------|
| `trapdata/api/utils.py` | ✅ Done | `get_http_session()` reads retry settings from Settings |
| `trapdata/api/datasets.py` | ✅ Done | Uses persistent sessions (could simplify retry param passing) |
| `trapdata/cli/worker.py` | ✅ Done | Registration uses Settings pattern, removed direct env var access |
| `trapdata/api/tests/antenna_api_server.py` | ✅ Done | Has registration endpoints |
| `trapdata/api/tests/test_worker.py` | ✅ Done | Has registration integration tests |
| `trapdata/api/tests/utils.py` | ✅ Done | Patches `Session.get/post` |

---

## Questions Resolved

1. **Should we use `requests.Session`?** → Yes, via `get_http_session()` (already implemented in PR #104)
2. **What retry strategy?** → Exponential backoff with urllib3.Retry (0.5s, 1s, 2s), 3 max retries
3. **Should image loading use auth session?** → No, separate session without auth for security
