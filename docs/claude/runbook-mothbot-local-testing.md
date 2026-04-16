# Runbook: Testing Mothbot Pipelines Locally Against Antenna

Exercises the `mothbot_insect_orders_2025` and `mothbot_panama_moths_2023`
pipelines end-to-end: ADC worker ↔ NATS ↔ Antenna Django ↔ Celery result
processor ↔ Postgres. This is the loop the author used while debugging the
YOLO-OBB integration.

## Prerequisites

- Local Antenna stack running: `antenna-django-1`, `antenna-celeryworker-1`,
  NATS, Postgres, Redis (checked with `docker ps`).
- ADC on branch `worktree-mothbot-pipeline` in
  `/home/michael/Projects/AMI/ami-data-companion/.claude/worktrees/mothbot-pipeline/`.
- `uv` installed; ADC's `.env` provides `AMI_ANTENNA_API_BASE_URL` and
  `AMI_ANTENNA_API_AUTH_TOKEN`.
- Project 20 on Antenna has at least one collection with source images
  (collection 10 = 16 starred images, collection 13 = 56 Panama images).

## 1. Bring up the ADC worker

The worker lives in tmux window 110. If it's already running, skip this.

```bash
# From the mothbot-pipeline worktree
cd /home/michael/Projects/AMI/ami-data-companion/.claude/worktrees/mothbot-pipeline
uv run ami worker
```

Watch for two GPU workers to spin up and `Checking for jobs for pipelines:`
logs listing every registered slug. The two mothbot slugs must be in that
list; otherwise Antenna will refuse to dispatch to them.

## 2. Register new pipelines with Antenna

Antenna polls the ADC `/info` endpoint (not the NATS worker). When you add
a new pipeline, you need to run the FastAPI server briefly so Antenna can
sync it.

### 2a. Start the ADC API server

```bash
# In a separate terminal (NOT tmux 110)
cd /home/michael/Projects/AMI/ami-data-companion/.claude/worktrees/mothbot-pipeline
uv run uvicorn trapdata.api.api:app --host 0.0.0.0 --port 5001 \
  > /tmp/ami_api.log 2>&1 &
# Give it ~20s to load every classifier's weights
sleep 20
curl -s http://localhost:5001/info | jq '.pipelines[].slug'
```

Confirm the new slug appears. The first startup will download classifier
weights; subsequent runs are cached under `~/.cache/torch/hub/models/`.

### 2b. Point Antenna's ProcessingService at the host API server

From inside the Django container, `localhost` is the container itself, not
your host. Use `host.docker.internal` instead (only needs to be set once
per Antenna DB):

```bash
docker exec antenna-django-1 python manage.py shell -c "
from ami.ml.models import ProcessingService
svc = ProcessingService.objects.get(name='BEAST1 (LINUXVISION)')
svc.endpoint_url = 'http://host.docker.internal:5001'
svc.save()
svc.create_pipelines()
"
```

`create_pipelines()` reads `/info` and upserts Pipeline + Algorithm rows.

### 2c. Known gotcha: stale pipeline metadata

`create_pipelines()` inserts new rows but does **not** rename existing
Pipelines or remove algorithm links that disappeared from `/info`. If you
rename an existing pipeline (e.g. the Mothbot insect-orders rename in
`1057b8a`), Antenna's row keeps the old name/description until you clean
it up manually:

```bash
docker exec antenna-django-1 python manage.py shell -c "
from ami.ml.models import Pipeline, Algorithm
p = Pipeline.objects.get(slug='mothbot_insect_orders_2025')
p.name = 'Mothbot YOLO + Insect Orders 2025'
p.description = '<new description from /info>'
p.save()
# Also drop any orphan algorithm links no longer produced by /info
stale = Algorithm.objects.filter(key='insect_order_classifier_mothbot_yolo_detector').first()
if stale:
    p.algorithms.remove(stale)
"
```

Skip this whole step for brand-new pipelines; only needed for renames.

### 2d. Stop the API server

```bash
pkill -f "uvicorn trapdata.api.api:app"
```

The ADC worker in tmux 110 handles actual processing via NATS; the HTTP
server is only needed for the `/info` sync.

## 3. Trigger an end-to-end job

Use Antenna's `test_ml_job_e2e` management command. `async_api` matches
production's NATS-dispatch path.

```bash
docker exec antenna-django-1 python manage.py test_ml_job_e2e \
  --project 20 --collection 13 \
  --pipeline mothbot_panama_moths_2023 \
  --dispatch-mode async_api
```

The command blocks and prints progress every 2s. Expected output:

```
✅ Job completed successfully
  Collect: 100.0% (SUCCESS) — Total Images: 55
  Process: 100.0% (SUCCESS) — Processed: 55, Failed: 0
  Results: 100.0% (SUCCESS) — Detections: 1015, Classifications: 1665
🔗 Job ID: <N>
```

For the insect-orders pipeline swap the slug; a smaller collection (10)
is fine to smoke-test.

If any images fail, the full Rich traceback shows up in the ADC worker
logs (tmux 110). The `Batch N failed during processing:` line is where
to start reading — local variables are included, which is invaluable for
bbox-shape bugs.

## 4. Verify detection quality

The job-complete output only shows counts. To sanity-check that detections
aren't garbage (e.g. full-image boxes), pull bbox stats from Postgres:

```bash
docker exec antenna-django-1 python manage.py shell -c "
from ami.main.models import Detection
from django.utils import timezone
import datetime
now = timezone.now()
recent = Detection.objects.filter(
    created_at__gte=now - datetime.timedelta(minutes=5),
    detection_algorithm__name__icontains='Mothbot',
)
sizes = [(d.bbox[2]-d.bbox[0], d.bbox[3]-d.bbox[1])
         for d in recent[:500] if d.bbox and len(d.bbox) == 4]
widths = sorted(s[0] for s in sizes)
heights = sorted(s[1] for s in sizes)
if widths:
    print(f'count={len(widths)}')
    print(f'w p10/p50/p90/max: {widths[len(widths)//10]:.0f}/{widths[len(widths)//2]:.0f}/{widths[len(widths)*9//10]:.0f}/{widths[-1]:.0f}')
    print(f'h p10/p50/p90/max: {heights[len(heights)//10]:.0f}/{heights[len(heights)//2]:.0f}/{heights[len(heights)*9//10]:.0f}/{heights[-1]:.0f}')
"
```

Healthy numbers on 3280×2464 source images: median width/height around
200–300 px, p90 under ~1000 px. If the median is near image width (e.g.
1200+ on a 3280-wide image) the detector is producing full-image boxes
— a red flag that the RGB/BGR channel order or another preprocessing
detail has regressed (see `0726b23`).

## 5. Iteration loop

After a code change:

1. Kill the worker: `Ctrl-C` in tmux 110.
2. If the change touches `/info` (pipeline list, descriptions, weights
   URLs, `detector_cls`), redo step 2.
3. Restart worker: `uv run ami worker` in tmux 110.
4. Re-trigger the job (step 3).

If the change only touches inference code paths (no new pipeline, no
algorithm metadata change), only steps 1, 3, and 4 are needed.

## Appendix: Which collection for which test

| Slug | Collection 10 (16 imgs, starred) | Collection 13 (56 imgs, Panama) |
|---|---|---|
| `mothbot_insect_orders_2025` | fastest smoke test | regression-size sanity check |
| `mothbot_panama_moths_2023` | fastest smoke test | realistic Panama load |

The Panama collection is the more realistic load — Panama diopsis images
triggered the YOLO-OBB negative-coord bug originally because the moths
frequently touch the image edge.
