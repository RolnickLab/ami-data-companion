# Runbook: Testing the YOLO Detector Pipelines Locally Against Antenna

Runs the Mothbot and AnyBug v2 pipelines end to end: ADC worker, Antenna
Django, NATS, the Celery result processor and Postgres. Use it after a change
to the YOLO detectors (`trapdata/ml/models/localization.py`) or to the worker
(`trapdata/antenna/worker.py`).

Pipeline slugs:

- `mothbot_insect_orders_2025`, `mothbot_panama_moths_2023`
- `anybug_mixed_v2_insect_orders_2025`, `anybug_mixed_v2_global_moths_2024`

## Prerequisites

- A local Antenna stack (Antenna `main`) with `django` and `celeryworker`
  running. Note the published Django port; this runbook uses `8000`.
- An Antenna user token for that stack.
- This branch checked out, with `uv sync` done.

## 1. Configure the worker

Put the settings in the ADC `.env` (or export them):

```bash
AMI_ANTENNA_API_BASE_URL=http://localhost:8000/api/v2
AMI_ANTENNA_API_AUTH_TOKEN=<token>
AMI_ANTENNA_SERVICE_NAME="Local YOLO test"
AMI_PIPELINES=anybug_mixed_v2_insect_orders_2025,anybug_mixed_v2_global_moths_2024
```

## 2. Register the pipelines

The worker uses async registration, so no ADC HTTP server is needed:

```bash
uv run ami worker register --project <project id>
```

Expect `created 2 pipelines` in the summary. The first run downloads the
weights into `~/.cache/torch/hub/models/` and checks their SHA-256.

## 3. Start the worker

```bash
uv run ami worker --project <project id>
```

Wait for `Checking for jobs for pipelines:` with the slugs from step 1.

## 4. Run a job

The project needs a collection of captures whose image URLs the worker can
fetch. For a quick set, serve `trapdata/tests/images` with
`python -m http.server` and create captures that point at it.

```bash
docker compose exec django python manage.py test_ml_job_e2e \
  --project <project id> --collection <collection id> \
  --pipeline anybug_mixed_v2_insect_orders_2025 --dispatch-mode async_api
```

The command blocks and prints progress. Expect every image processed and no
failures. Worker errors appear in the worker log, starting at the line
`Batch N failed during processing:`.

The job summary counts can be lower than the rows in the database. To check
results, count `Detection` and `Classification` rows for the project.

## 5. Check detection quality

Look for these signs of a regression:

- **Full-image boxes.** A median box width close to the image width means the
  RGB/BGR channel order is wrong (see `_tensor_to_bgr_numpy`).
- **Different counts for the same image in two runs.** Batches of mixed image
  sizes change the YOLO letterbox. `predict_batch` groups images by size to
  prevent this.
- **Empty crops.** Boxes at the image edge must be clamped
  (see `_corners_to_yolo_detection`).

## 6. Iterate

After a code change, restart the worker and run step 4 again. Run step 2 again
only when the pipeline list, names, descriptions or weights change.
