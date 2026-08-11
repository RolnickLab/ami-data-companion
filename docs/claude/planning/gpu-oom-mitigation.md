# GPU Out-of-Memory Mitigation for the Antenna Worker

Status: implemented on branch `fix/gpu-oom-mitigation` (August 2026).

## Problem

In a production deployment (August 2026), ML jobs failed with hundreds of
`CUDA out of memory` errors plus a few `CUBLAS_STATUS_ALLOC_FAILED` errors.
The deployment shape: one 24 GiB GPU shared by four worker processes serving
multiple environments. Measured from the error logs:

- Failed allocations were consistently 426 MiB (504 occurrences) or 736 MiB
  (8 occurrences) — the same allocation retried batch after batch, not random
  sizes. Each failed batch is caught in `_process_batch` and reported to
  Antenna as per-image task errors, so the job keeps running and keeps
  failing on every subsequent batch.
- Free memory at failure ranged 61–457 MiB; the failing process held
  ~14.5 GiB total, of which 13.49 GiB was allocated by PyTorch.
- Reserved-but-unallocated memory was consistently 757–899 MiB
  (fragmentation, since the allocator could not satisfy a 426 MiB request
  despite ~800 MiB reserved).
- A neighbouring process held 6.12 GiB at the same time; two others held
  ~125 MiB each (CUDA context only, effectively idle).

## Diagnosis: peak working set, not a leak

Everything below is established by code reading unless labelled measured.

### The GPU batch size is governed by the wrong setting

On the Antenna worker path, the size of every GPU forward pass is
`antenna_api_batch_size` (default 24), not the GPU batch-size settings:

- `get_rest_dataloader` (`trapdata/antenna/datasets.py:427`) sizes the
  `RESTDataset` fetch with `settings.antenna_api_batch_size`, and each
  yielded batch is the entire API fetch, pre-collated
  (`datasets.py:317-320`).
- `_process_batch` runs the whole batch of full-resolution (~4K) images
  through FasterRCNN in a single forward call
  (`trapdata/antenna/worker.py:254`). Neither
  `InferenceBaseClass.predict_batch` (`trapdata/ml/models/base.py:276`) nor
  `ObjectDetector.predict_batch` (`trapdata/ml/models/localization.py:163`)
  chunks its input.
- All crops from all detections in the batch are stacked into one tensor and
  classified in a single forward call (`worker.py:321-322` for the terminal
  classifier, `worker.py:176-177` for the binary filter). The detector allows
  up to 500 detections per image (`ml/models/localization.py:255`), so this
  stack is unbounded in practice.
- `localization_batch_size` (default 8, described in settings as "reduce this
  if you run out of memory") and `classification_batch_size` (default 20) are
  never referenced under `trapdata/antenna/`. The synchronous FastAPI path
  does pass them (`trapdata/api/api.py:223,239,279`); the worker path was
  built without them. Even `benchmark.py:73` sets
  `settings.localization_batch_size` from its `--gpu-batch-size` flag, but
  nothing on the worker path ever reads it — the knob is a silent no-op.

### What the peak is made of (rough estimates from code reading)

- Current batch of 24 full-resolution images as float32 tensors:
  ~2.4 GiB (24 × 3 × 2160 × 4096 × 4 B).
- `CUDAPrefetcher` (`datasets.py:445`) holds the *next* full batch on the GPU
  while the current one is processed — roughly another 2.4 GiB.
- `image_tensors` (`worker.py:275`) keeps all full-resolution GPU tensors
  alive through detection *and* classification (needed for crop slicing).
- FasterRCNN activations for a 24-image forward pass (internally resized to
  ≤1333 px): several GiB.
- Model weights for detector + binary filter + terminal classifier: ~1–2 GiB.

These estimates are consistent with the measured 13.49 GiB PyTorch-allocated
peak. With four processes sharing 24 GiB, the card is exhausted whenever more
than one process is mid-batch at the same time — which also explains why the
same configuration often works: a single busy process fits.

### Why this is not a leak

- Per-batch intermediates are local to `_process_batch` (`worker.py:203`, the
  docstring states this deliberately) and `torch.cuda.empty_cache()` runs
  after every batch (`worker.py:395`).
- Models are constructed per job, lazily on the first batch
  (`worker.py:464-478`), and are function locals released by reference
  counting when `_process_job` returns. There is no cross-job model cache.
- An existing regression test (`trapdata/antenna/tests/test_memory_leak.py`)
  pins host-RSS stability across batches. A prior analysis of host-RAM
  blowup (DataLoader `pin_memory` × `prefetch_factor`, `datasets.py:439-441`)
  concerns host memory in the DataLoader subprocesses, which never touch
  CUDA — a separate problem from this one.

One cross-job gap does exist: nothing releases cached allocator blocks when a
job *ends*. `empty_cache()` runs per batch and at the start of the *next*
claimed job (`worker.py:436`), so a process idling between jobs retains
reserved VRAM (roughly its freed model weights plus remnants) that co-tenant
processes on the shared card cannot use.

### Allocator configuration

No `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF` is set anywhere in the
repo, and `torch.cuda.set_per_process_memory_fraction` is not used. The
measured 757–899 MiB reserved-but-unallocated at failure is the fragmentation
signature that `expandable_segments:True` targets (the OOM message itself
recommends it).

## Fix

Smallest changes that address the dominant cause, in order of impact:

1. **Honor the GPU batch-size settings on the worker path.** Construct the
   worker's models with `batch_size` from settings (mirroring the FastAPI
   path): detector gets `localization_batch_size`, binary filter and terminal
   classifier get `classification_batch_size`. Add a chunked-inference helper
   in `worker.py` that runs `predict_batch` + `post_process_batch` over the
   input in chunks of `model.batch_size`, so the forward-pass peak is capped
   regardless of `antenna_api_batch_size`. Results are unchanged: detection
   and per-crop classification are independent per item, and softmax is
   per-row. (For mixed-size image batches, sub-batching can change FasterRCNN's
   internal padding, which can perturb detections near image borders
   negligibly.)
2. **Bounded adaptive backoff.** If a chunk still hits CUDA OOM (a co-tenant
   process spiking), the helper halves the chunk size and retries, down to a
   chunk of 1, calling `empty_cache()` between attempts. The reduction lasts
   for the remainder of that call; the next batch starts fresh from the
   configured size, so transient neighbour pressure shrinks chunks only while
   it persists.
3. **Release GPU memory at the end of every job.** In `_process_job`'s
   `finally` block, drop references to the models, prefetcher, and loader,
   then call `empty_cache()`, so an idle process returns cached VRAM to the
   shared card instead of holding it until its next job.
4. **Default the allocator to `expandable_segments:True`.** Set both
   `PYTORCH_ALLOC_CONF` (newer name) and `PYTORCH_CUDA_ALLOC_CONF` at worker
   startup, only when the operator has set neither, so deployments can still
   override or disable it via the environment.
5. **Update stale documentation** in `datasets.py` (which currently says the
   async worker uses `antenna_api_batch_size` for the GPU batch) and the
   worker-tuning notes.

After this change `antenna_api_batch_size` controls only fetch granularity
and how many downloaded images are resident per batch — no longer the
forward-pass size. Operators can additionally lower it to shrink the resident
image tensors and the prefetcher's double-buffer.

## Deliberately not changed

- **`CUDAPrefetcher` double-buffering** — a throughput feature with a bounded
  cost (one extra batch of images); lowering `antenna_api_batch_size` shrinks
  it without a code change.
- **Per-batch `empty_cache()`** (`worker.py:395`) — slightly hurts throughput
  but is polite on a shared card; not the target of this fix.
- **`set_per_process_memory_fraction`** — would relabel the failure (OOM at
  the cap) without reducing demand, and caps legitimate bursts when the card
  is otherwise idle.
- **Cross-job model caching** — would avoid per-job weight reloads but pins
  VRAM in idle processes, the opposite of what a shared card needs.
- **Default values of the batch-size settings** — 8 (localization) and
  20 (classification) are the long-standing defaults of the synchronous path.

## What still needs verification (no GPU available in this environment)

- That the chunked detector/classifier forward passes produce identical
  results on a real GPU end-to-end run (CPU-path tests pass; the CUDA
  prefetcher path is not exercised without a GPU).
- The actual post-fix peak VRAM per busy process (estimated, not measured:
  roughly 5–9 GiB with defaults) — observe `nvidia-smi` during a real job.
- That `expandable_segments` behaves on the deployed driver/vGPU combination
  (PyTorch falls back with a warning where unsupported).
- Whether the backoff path ever triggers in steady state (its warning log is
  the signal that the card is still over-committed and process count or batch
  sizes need ops-side tuning).

## Validation protocol for production

1. Deploy to one host; leave the rest unchanged as control.
2. During a real job, watch per-process GPU memory (`nvidia-smi`) — expect
   busy-process peaks well below the previous ~14.5 GiB, and idle processes
   dropping to near context-only after a job completes.
3. Grep worker logs for `CUDA out of memory` recurrence and for the new
   chunk-size reduction warnings.
4. Compare seconds-per-image from the per-batch log line before and after —
   chunking the detector forward is expected to be roughly throughput-neutral
   since the GPU already serializes the work internally.
