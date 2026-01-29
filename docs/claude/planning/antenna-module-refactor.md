# Refactor: Create `trapdata/antenna/` Module

**PR:** #94 (carlosg/pulldl branch)
**Author:** mihow
**Decision:** Rewrite commit history since worker is new code introduced in this PR

## Goal

Extract Antenna platform integration code from `trapdata/cli/worker.py` into a dedicated `trapdata/antenna/` module to:
1. Separate business logic from CLI concerns
2. Enable reuse in a future standalone worker app
3. Provide a home for upcoming Antenna export functionality

## Current State

`trapdata/cli/worker.py` (508 lines) contains:
- Antenna API client logic (fetching jobs, posting results, fetching projects)
- Pipeline registration logic
- Worker loop and job processing orchestration
- ML pipeline calls

Other files with Antenna-related code:
- `trapdata/api/schemas.py` - Pydantic models for Antenna API requests/responses
- `trapdata/api/datasets.py` - `RESTDataset` that streams tasks from Antenna
- `trapdata/api/utils.py` - `get_http_session()` with retry logic

## Target Structure

```
trapdata/antenna/
├── __init__.py          # Public API exports
├── client.py            # Antenna API client (jobs, results, projects)
├── worker.py            # Worker loop + job processing logic
├── registration.py      # Pipeline registration with projects
├── schemas.py           # Antenna-specific Pydantic models (moved from api/schemas.py)
└── datasets.py          # RESTDataset (moved from api/datasets.py)

trapdata/cli/
└── worker.py            # Thin wrapper: ~30 lines, just CLI arg parsing
```

## Refactor Steps

### Step 1: Create module structure

Create `trapdata/antenna/__init__.py` with public exports.

### Step 2: Move Antenna schemas

Move Antenna-specific models from `trapdata/api/schemas.py` to `trapdata/antenna/schemas.py`:
- `AntennaJob`
- `AntennaJobsListResponse`
- `AntennaTask`
- `AntennaTasksResponse`
- `AntennaTaskResult`
- `AntennaTaskResultError`
- `AsyncPipelineRegistrationRequest`
- `AsyncPipelineRegistrationResponse`

Keep in `trapdata/api/schemas.py` (used by FastAPI):
- `SourceImageInput`, `SourceImageResponse`
- `DetectionResponse`, `ClassificationResponse`
- `PipelineRequest`, `PipelineResultsResponse`
- `ServiceInfoResponse`, `PipelineInfoResponse`

### Step 3: Move RESTDataset

Move `RESTDataset` and `get_rest_dataloader()` from `trapdata/api/datasets.py` to `trapdata/antenna/datasets.py`.

Update imports in `trapdata/cli/worker.py`.

### Step 4: Create client.py

Extract from `trapdata/cli/worker.py`:
- `_get_jobs()` → `antenna/client.py:get_jobs()`
- `post_batch_results()` → `antenna/client.py:post_batch_results()`
- `get_user_projects()` → `antenna/client.py:get_user_projects()`

### Step 5: Create registration.py

Extract from `trapdata/cli/worker.py`:
- `register_pipelines_for_project()` → `antenna/registration.py`
- `register_pipelines()` → `antenna/registration.py`

### Step 6: Create worker.py

Extract from `trapdata/cli/worker.py`:
- `run_worker()` → `antenna/worker.py`
- `_process_job()` → `antenna/worker.py`
- `SLEEP_TIME_SECONDS` constant

### Step 7: Slim down CLI wrapper

Reduce `trapdata/cli/worker.py` to thin CLI wrapper:
```python
"""CLI commands for Antenna worker."""
import typer
from trapdata.antenna.worker import run_worker
from trapdata.antenna.registration import register_pipelines

# Typer command definitions only, no business logic
```

### Step 8: Update imports

Update all files that import from moved locations:
- `trapdata/cli/base.py` - imports worker commands
- `trapdata/api/tests/test_worker.py` - imports worker functions
- Any other files importing Antenna schemas

### Step 9: Run tests

```bash
pytest trapdata/api/tests/test_worker.py
ami test all
```

## Files Changed

| File | Action |
|------|--------|
| `trapdata/antenna/__init__.py` | Create |
| `trapdata/antenna/client.py` | Create |
| `trapdata/antenna/worker.py` | Create |
| `trapdata/antenna/registration.py` | Create |
| `trapdata/antenna/schemas.py` | Create (move from api/schemas.py) |
| `trapdata/antenna/datasets.py` | Create (move from api/datasets.py) |
| `trapdata/cli/worker.py` | Slim down to CLI wrapper |
| `trapdata/api/schemas.py` | Remove Antenna-specific models |
| `trapdata/api/datasets.py` | Remove or delete if empty |
| `trapdata/cli/base.py` | Update imports |
| `trapdata/api/tests/test_worker.py` | Update imports |

## Notes

- `trapdata/api/utils.py` (`get_http_session`) stays in `api/` since it's generic HTTP utility
- Future Antenna export PR can add `trapdata/antenna/export.py`
- This refactor is purely structural - no behavior changes

## Risks

### High Risk
1. **Circular imports** - `antenna/worker.py` imports from `api/api.py` which might import schemas. Check import order carefully.
2. **Schema dependencies** - Some schemas in `api/schemas.py` (e.g., `DetectionResponse`, `PipelineResultsResponse`) are used by both FastAPI and Antenna. Don't move these - only move Antenna-specific ones.
3. **Broken CLI registration** - Typer commands must be properly wired in `cli/base.py`. If `app.command()` decorators aren't set up right, commands silently disappear.

### Medium Risk
4. **Missing imports** - Easy to miss an import somewhere. A file might work in isolation but fail when the full app loads.
5. **Test imports** - `test_worker.py` imports worker functions directly. Must update.
6. **`__init__.py` exports** - If `trapdata/antenna/__init__.py` doesn't export the right things, imports like `from trapdata.antenna import run_worker` fail.

### Low Risk
7. **Relative vs absolute imports** - Prefer absolute imports (`from trapdata.antenna.client import ...`) for clarity.

## Validation Checklist

Run these checks after each major step, not just at the end:

```bash
# 1. Check module imports work (no circular import errors)
python -c "from trapdata.antenna import client, worker, registration, schemas, datasets"

# 2. Check CLI commands are registered
ami worker --help
ami register --help

# 3. Check no old imports remain
grep -rn "from trapdata.cli.worker import" trapdata/ --include="*.py"
grep -rn "from trapdata.api.schemas import Antenna" trapdata/ --include="*.py"
grep -rn "from trapdata.api.datasets import REST" trapdata/ --include="*.py"

# 4. Run the specific worker tests
pytest trapdata/api/tests/test_worker.py -v

# 5. Run full test suite
pytest

# 6. Check for type/import errors without running tests
python -c "import trapdata.cli.base"
python -c "import trapdata.api.api"

# 7. Linting (catches unused imports, etc.)
flake8 trapdata/antenna/ trapdata/cli/worker.py trapdata/api/schemas.py
```

### Integration Test (if possible)

```bash
# Start mock Antenna server (from tests)
python -m trapdata.api.tests.antenna_api_server &

# Try worker against it
ami worker --pipeline moth_binary
```

## Common Mistakes to Avoid

1. **Don't move `DetectionResponse`, `PipelineResultsResponse`, etc.** - These are used by FastAPI routes, not just Antenna
2. **Don't forget `__init__.py`** - Every new directory needs one
3. **Don't leave dead imports** - After moving code, remove old imports from source files
4. **Don't mix refactor with fixes** - If you find bugs, note them but don't fix in same commit
5. **Check `api/datasets.py` after moving** - If it's empty or only has unused code, delete it entirely rather than leaving a stub

## Git Workflow

Since this is new code being introduced in PR #94, rewrite history to place code in the correct location from the start.

After refactor is complete and tests pass:

```bash
# Interactive rebase to reorganize commits
git rebase -i main

# Suggested final commit structure:
# 1. "Add Antenna module for platform integration"
#    - trapdata/antenna/ module with client, worker, registration, schemas, datasets
# 2. "Add CLI commands for Antenna worker"
#    - Thin cli/worker.py wrapper
# 3. "Add worker tests and configuration"
#    - Tests, settings, .env.example updates

# Force push (safe since we own the branch)
git push --force-with-lease
```

## Execution Checklist

- [ ] Create `trapdata/antenna/__init__.py`
- [ ] Create `trapdata/antenna/schemas.py` (move Antenna models from api/schemas.py)
- [ ] Create `trapdata/antenna/datasets.py` (move RESTDataset from api/datasets.py)
- [ ] Create `trapdata/antenna/client.py` (extract from cli/worker.py)
- [ ] Create `trapdata/antenna/registration.py` (extract from cli/worker.py)
- [ ] Create `trapdata/antenna/worker.py` (extract from cli/worker.py)
- [ ] Slim down `trapdata/cli/worker.py` to CLI wrapper
- [ ] Update `trapdata/api/schemas.py` (remove moved models)
- [ ] Update `trapdata/api/datasets.py` (remove moved code or delete)
- [ ] Update imports in `trapdata/cli/base.py`
- [ ] Update imports in `trapdata/api/tests/test_worker.py`
- [ ] Run `pytest trapdata/api/tests/test_worker.py`
- [ ] Run `ami test all`
- [ ] Run `black trapdata/ && isort trapdata/`
- [ ] Interactive rebase to clean history
- [ ] Force push
