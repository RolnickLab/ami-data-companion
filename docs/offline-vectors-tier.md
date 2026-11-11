# Offline iNat/GBIF vectors: the first training-data tier

Michael's plan has three optional data tiers for retraining a classifier head:

1. **Offline iNat/GBIF data with precomputed vectors.** The processing service holds this itself.
2. All verified data across Antenna projects.
3. The user's own project occurrences. This is what `/train` does today.

The minimum input is a **taxa list**. This doc covers tier 1. It also covers how tier 1 sits next to tier 3.

The backbone is frozen. So each image is embedded **once per backbone**. After that, a head is fitted
on vectors alone. No image or GPU is needed at train time.

## 1. Where the data lives today (what I actually found)

| What | Where | Notes |
|---|---|---|
| Image index | BigQuery `leps-ai.global_all_leps_2605.training_images` | 23,890,349 rows, 7.3 GB logical. Columns: `dataset_source_uuid, source, photo_id, inat_taxon_id, gbif_id, absolute_url, relative_local_path, fetch_status, image_width/height/size, corrupted, life_stage, license, attributes, created_at`. |
| Image counts (queried 2026-10-29) | same table, `fetch_status` | 23,889,543 `downloaded` over **59,542** species; 803 failed; 3 corrupted. |
| Per-species spread | same | Quartiles of images per species: min 1, 4, 14, 67, max 493,808. 34,049 species have 10 or more images. |
| Species names | `leps-ai.global_all_leps_2605.inat_taxa` | Joined on `inat_taxon_id` (`species_name`, `family`). |
| Regional taxa lists | `leps-ai.global_all_leps_2605.taxa_regions` (756k rows) + `regions` (33.5k) | `species_id -> region_gid`. This is the natural source of a taxa list. |
| Model registry (scaffolded) | `track_models`, `track_experiments`, `track_experiment_results` | 1 row each. Not used here. |
| Image bytes | Arbutus S3 bucket `ami-trainingdata`, squashfs shards `task_N.sqfs`, mounted on the GPU workspace at `/mnt/squash-0..9` | Those shards are the **butterfly** set (shard = `photo_id % 10`). On 2026-10-30 the mounts gave I/O errors (`/mnt/s3` mountpoint-s3 broken). `global_all_leps_2605` paths look like `global_all_leps_2605/task_49/589/<uuid>.jpg`, which is not in those shards. Every row also has `absolute_url` (iNat open-data S3, public), and that is what the demo used. |
| Earlier, smaller dataset | `leps-ai.global_butterflies_2604.*` | Butterflies only (~7k species). The distillation experiments used it. |
| Pipeline code that made manifests and vectors | `~/antenna-models-training/bioclip-distill-leps/src/step1_global.py` (BigQuery manifest, observation-grouped 80/10/10 split on `gbif_id`), `src/step3_embed.py` (open_clip, L2-normalised, float16, written to Parquet) | Local copy holds **manifests only**: `data/manifest.parquet` (10k rows), `manifest_b.parquet` (5k), `manifest_d.parquet` (44k). No vectors. |
| Existing vectors (from memory notes, **not re-checked**: no access from here) | GPU VM `/mnt/melabbas/emb25h_subset/` (BioCLIP-2.5 ViT-H/14, 1024-d, ~331k images = 257,779 train + 73,242 test over 6,687 butterfly species, cap 50/species), and `data/embeddings_global.parquet` (BioCLIP-2, 768-d) | `/mnt` on that VM is ephemeral. A backup of `/mnt` was taken on 2026-09-03 to `s3://ami-trainingdata/ai-for-leps/backup/2026-09-03_ami-workspace-02-gpu_mnt-ephemeral/`. **Butterflies only.** The moth heads (Newfoundland, Panama) need `global_all_leps_2605`, which has no vectors yet. |

**Bottom line:** the image index and the images exist. There are **no vectors for the full leps set**.
The only 1024-d (BioCLIP-2.5) vectors are a butterfly subset, and they are on an ephemeral disk.
Nothing in BigQuery holds vectors.

## 2. Storage format (recommendation)

**Parquet on object storage, partitioned by backbone, then by taxon.**

```
<root>/<backbone slug>/                        e.g. hf-hub_imageomics_bioclip-2.5-vith14/
    _manifest.json        backbone_key, dimensions, dtype=float16, l2_normalised, updated_at
    _taxa.parquet         taxon_key, taxon_name, rows        (one line per species)
    taxon_key=<inat_taxon_id>/part-<run>.parquet
```

Columns in each part file:

| column | type | why |
|---|---|---|
| `taxon_key` | string | `inat_taxon_id`; stable across renames |
| `taxon_name` | string | what Antenna sends today |
| `image_id` | string | `dataset_source_uuid`, joins back to BigQuery |
| `image_url` | string | `absolute_url`, for audit and re-embedding |
| `split` | string | `train` / `val` / `test`, grouped by observation (`gbif_id`) |
| `backbone_key` | string | belt and braces; also in the path |
| `vector` | fixed_size_list<float16>[1024] | L2-normalised, same as the serving path |

Why this shape:

- **The backbone is in the path.** Vectors from two backbones live in different spaces. A head fitted
  on one is useless on the other. They must never mix.
- **One directory per taxon.** A run for 300 species opens about 300 small files. It never scans the rest.
  No query engine is needed.
- **float16.** It is what Antenna already stores and sends. The head is linear over unit vectors, so
  float16 loses nothing that matters.
- **Append-only part files.** Shards and reruns add files and never rewrite old ones. `_taxa.parquet` is
  rebuilt from file footers.

Why not the alternatives:

| Option | Problem |
|---|---|
| BigQuery | Every `/train` would be a billed scan. The service would need GCP credentials. It is fine as a *catalog*: the Parquet can be attached as an external table later. |
| pgvector in Antenna | Adds 3 to 50 GB to the production database for data that is not project data. |
| One big `.npz` | Must be read whole. Cannot be appended to or sharded. |

## 3. Estimated size

Per row: 1024 x 2 bytes = **2,048 B** for the vector. About 80 B for the other columns
(dictionary-encoded strings). float16 vectors barely compress, so I count about **2.1 KB per row**.

Image counts come from the BigQuery query above. "Cap N" means at most N images per species.

| Set | Images | Vectors only | With metadata (about) |
|---|---|---|---|
| Cap 50 / species | 1,345,071 | 2.75 GB | **~2.9 GB** |
| Cap 100 / species | 2,058,070 | 4.21 GB | **~4.4 GB** |
| Cap 500 / species | 5,026,784 | 10.29 GB | **~10.7 GB** |
| Everything | 23,889,543 | 48.9 GB | **~50 GB** |

Arithmetic: e.g. 5,026,784 x 2,048 B = 10.29 GB, plus 5.03 M x 80 B = 0.40 GB.

**Recommendation: start with cap 100** (about 4.4 GB). The distillation work showed the head gains
little past about 50 images per species. Even 100 per species is small enough to keep a full local
copy on the service's disk.

## 4. How `/train` loads "vectors for these species"

Implemented on this branch:

- `trapdata/ml/offline_vectors.py`: `OfflineVectorSource(root).load(species, backbone_key, max_per_species, all_as_train)`
  1. Reads `_taxa.parquet` (about 60k rows, about 2 MB) to map names to keys. Matching ignores case and extra spaces.
  2. Reads only `taxon_key=<k>/*.parquet` for the matched species, with only the columns it needs.
  3. Returns `TrainingRow`s (`source="offline"`) and a list of **missing species**.
- Setting `AMI_OFFLINE_VECTORS_DIR` (`Settings.offline_vectors_dir`). Unset means requests that ask
  for offline data are refused with a 422.
- `TrainRequest` additions. All are optional. The defaults keep the old behaviour.
  - `use_offline_inat: bool = False`. Same name as the Antenna job switch.
  - `classes: list[str] | None`. The taxa list. It is **required** when there is no `dataset_url`.
  - `backbone_key: str | None`. Defaults to the `backbone_name` of the head being retrained.
  - `offline_max_per_species: int | None`. Caps rows per species and split.
  - `dataset_url` is now optional. A request must have a `dataset_url`, or `use_offline_inat` with `classes`.
- `TrainResponse.offline`: rows, splits, species found, per-species counts, `missing_species`. Missing
  species also show up in `warnings`.

The modes:

| Request | Train rows | Held-out (test) rows |
|---|---|---|
| `dataset_url` only (today) | Antenna | Antenna |
| `dataset_url` + `use_offline_inat` | Antenna + **all** offline rows | **Antenna only**. The comparison stays on the user's own crops. |
| `use_offline_inat` + `classes`, no dataset | offline `train` | offline `test` + `val` |

Speed: a 500-species regional list at cap 100 means at most 50k rows, or about 105 MB in about 500 files.
From local disk that takes about a second. Over S3 it is about 500 small GETs, which is seconds if done
in parallel. Fitting stays the existing linear head over the same matrices.

Serving the files: the service reads a **local directory**. In production that directory is either:

- (a) a copy synced from the bucket (`aws s3 sync s3://<bucket>/offline-vectors/<backbone>/ $AMI_OFFLINE_VECTORS_DIR/<backbone>/`). This is simplest. At cap 100 it is about 4.4 GB.
- (b) a `mountpoint-s3` mount, the same thing already used for the squashfs shards on Arbutus.

No code change is needed for either. Reading `s3://` directly through `pyarrow.fs` is a small later
addition if needed.

## 5. Precomputing the vectors

Script: `scripts/precompute_offline_vectors.py`

- Input is either `--manifest` (CSV or Parquet: `taxon_key, taxon_name, image_id, split, image_path|image_url`)
  or `--image-dir` (one folder per species).
- It embeds with `open_clip.create_model_and_transforms(<backbone>)`, the same loader as
  `trapdata/ml/models/bioclip.py`, and L2-normalises. Unreadable images are skipped and counted.
- It writes part files, updates `_manifest.json` and rebuilds `_taxa.parquet`.
- `--shard i --num-shards n` splits the job across machines.
- It is tested on generated images with a fake encoder. It has **not** been run with the real model.

Outline of the production run:

1. Manifest from BigQuery. Reuse `step1_global.py` on `global_all_leps_2605`, joined with `inat_taxa`,
   using `fetch_status='downloaded'` and the `gbif_id`-grouped split. Cap per species and split. Point
   `image_path` at a local copy of the images, or use `image_url` (`absolute_url`).
   Always `--dry_run` first. Selecting `absolute_url` and `relative_local_path` scans about 5.5 GB.
   Counting by taxon alone scans about 0.5 GB.
2. Run the script on the Arbutus GPU workspace in shards. Write to persistent disk (`/data`), not `/mnt`.
3. `aws s3 sync` the backbone directory to the object store.
4. Optional quick win: convert the existing butterfly `emb25h_subset` vectors into this layout, if the
   backup still holds them.

GPU time. **Every figure here is an estimate.**

- Measured before: BioCLIP-2.5 ViT-H/14 ran at **~11 img/s** on a MIG `H100L-1-24C` slice. That run
  was limited by cold, scattered squashfuse reads, not by the GPU.
- At that rate: cap 50 takes ~34 h, **cap 100 takes ~52 h**, cap 500 takes ~127 h (~5 days), and
  everything takes ~600 h (~25 days).
- If the reads were sequential (sort the manifest by shard and path, use more DataLoader workers), the
  GPU would be the limit. A rough figure for ViT-H/14 at 224 px is ~150-300 img/s on that slice
  (~170 GFLOPs per image). Then cap 100 takes ~2-4 h. **Not measured.**

## 6. What Antenna has to change to turn on `use_offline_inat`

1. `ami/ml/training/config.py`: remove the `raise` that refuses `use_offline_inat`.
2. `ami/ml/training/service.py::send_training_request`: send `use_offline_inat`, and send `classes`
   (the taxa list names). Optionally send `offline_max_per_species` and `backbone_key`.
3. When `use_project_occurrences` is off, **do not build a dataset.** Today `build_dataset` raises
   `NotEnoughVerifiedData` when there are no train or test rows. Send `dataset_url=None` with `classes` instead.
4. The training-result callback: `dataset_url` may be `null`. Store `result.offline` with the version,
   because it records what the head learned from.
5. Make the Antenna-side `TrainingRequest` schema match: optional `dataset_url`, plus the new fields.
6. Optional: only offer the switch when the service says it has offline vectors. A flag in `/info`
   would do it, but this branch does not add one yet.

## 7. Open questions for Michael

1. **Which store and which bucket?** `ami-trainingdata` (old endpoint) or a new prefix on
   `object-arbutus.alliancecan.ca`? And should the service sync a full copy or mount it?
2. **How many images per species?** Cap 100 (~4.4 GB, ~52 h at measured I/O) or cap 500 (~10.7 GB)?
3. **Taxon matching.** Antenna sends names. Synonyms and renames will miss. Could Antenna send GBIF or
   iNat taxon keys with the names? The layout is already keyed by `inat_taxon_id`.
4. **Weighting.** Offline iNat photos can outnumber a project's trap crops 100 to 1, and they look
   different (field photos versus trap crops). Should Antenna rows get more weight, or offline rows a
   lower cap, when both are used?
5. **Offline-only evaluation.** With no Antenna rows, the comparison runs on iNat test photos. The
   incumbent may have been trained on the same iNat pool. The observation-grouped split avoids photo
   leakage only if the same split rule was used. Is that good enough to promote on, or should
   offline-only heads never auto-promote?
6. **Species with no data anywhere.** A species in the taxa list with no offline vectors, no Antenna
   rows and no incumbent weights gets an untrained column. Today it is only a warning. Should it be
   dropped from the head instead?
7. **Tier 2** (verified data across all Antenna projects) could ship to the service in this same layout,
   with `source="antenna"`. Worth doing?

## 8. Demo set built on 2026-10-30

- Location: `~/offline-vectors-demo/hf-hub_imageomics_bioclip-2.5-vith14/` (4.1 MB).
- Contents: 20 Newfoundland species x 100 images = 2,000 vectors. Split by `gbif_id`: 1,637 train, 363 test+val.
- How it was made:
  - The images came from `absolute_url`, downloaded on the GPU workspace (2,598 of 2,600 succeeded).
  - They were embedded with this script in fp32, using the `open_clip` preprocess and L2 normalisation, which is the same as `BioCLIPWithLinearHead`.
  - It took 3 min 23 s on the H100 MIG slice.
- Check: the deployed Newfoundland head (749 classes) gets top-1 0.953 on these vectors. So they are in the space the head expects.
- `/train` with `use_offline_inat` and the 20 names, and no dataset: new head 0.983 vs current 0.945 on the 363 held-out rows.
  - This is **not** a fair comparison. The new head picks among 20 species; the current head picks among 749.
  - The current head may also have been trained on these same iNat photos.
