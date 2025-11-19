# CLAUDE.md - AI Agent Development Guide

This file helps AI agents (like Claude) work efficiently with the AMI Data Companion codebase.

## 🚨 IMPORTANT - Cost Optimization

**Every call to the AI model API incurs a cost and requires electricity. Be smart and make as few requests as possible. Each request gets subsequently more expensive as the context increases.**

### Efficient Development Practices

1. **Learn from this file** - Add learnings and gotchas to avoid repeating mistakes and trial & error
2. **Ignore line length and type errors until the very end** - Use command line tools to fix those (black, flake8)
3. **Always prefer command line tools** to avoid expensive API requests (e.g., use git and jq instead of reading whole files)
4. **Use bulk operations and prefetch patterns** to minimize database queries
5. **Commit often** - Small, focused commits make debugging easier
6. **Use TDD whenever possible** - Tests prevent regressions and document expected behavior
7. **Keep it simple** - Always think hard and evaluate more complex approaches and alternative approaches before moving forward

### Think Holistically

**What is the PURPOSE of this tool?** Why is it failing on this issue? Is this a symptom of a larger architectural problem? Take a step back and analyze the root cause.

---

## 📊 Repository Overview

**AMI Data Companion** is a desktop application for analyzing images from autonomous insect monitoring stations using deep learning models.

### Purpose
- Process images from automated insect monitoring stations (AMI)
- Detect insects/moths using object detection models (FasterRCNN)
- Classify detected objects to species level
- Track individual organisms across multiple frames
- Export structured data about detected species and occurrences

### Key Stats
- **Package Name:** trapdata (version 0.6.0)
- **Python Version:** 3.10+
- **Total Python Files:** 68 files
- **Total Lines of Code:** ~6,164 lines
- **License:** MIT
- **Repository:** https://github.com/RolnickLab/ami-data-companion

---

## 🏗️ Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Entry Points                            │
│  - CLI (Typer): ami command                                 │
│  - GUI (Kivy): Desktop application                          │
│  - API (FastAPI): REST API for remote inference             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   ML Pipeline (trapdata.ml)                 │
│  1. Localization (FasterRCNN) → Detect insects              │
│  2. Binary Classification → moth/non-moth filter            │
│  3. Species Classification → Identify species               │
│  4. Feature Extraction → CNN features for tracking          │
│  5. Tracking → Link detections across frames                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                Database Layer (trapdata.db)                 │
│  - MonitoringSession: Capture periods (e.g., one night)     │
│  - TrapImage: Source images from camera traps               │
│  - DetectedObject: Detected insects with classifications    │
│  - Occurrences: Tracked individuals across frames           │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Camera Trap Images
    ↓
Import (scan filesystem, create DB records)
    ↓
Queue Management (add to processing queue)
    ↓
ML Pipeline:
  1. Localization → bounding boxes
  2. Binary Classification → filter non-moths
  3. Species Classification → identify species
  4. Feature Extraction → get CNN embeddings
  5. Tracking → match across frames
    ↓
Export (JSON, CSV, with cropped images)
```

### Technology Stack

- **ML Framework:** PyTorch + torchvision + timm
- **Web API:** FastAPI + uvicorn
- **Database:** SQLAlchemy (SQLite/PostgreSQL)
- **GUI:** Kivy
- **CLI:** Typer + Rich
- **Validation:** Pydantic
- **Testing:** pytest
- **Package Management:** Poetry

---

## 📁 Directory Structure Quick Reference

```
trapdata/
├── api/              # FastAPI web service (POST /process, GET /info)
├── cli/              # Typer CLI commands (import, run, export, queue, show)
├── common/           # Shared utilities (file management, logging, schemas)
├── db/               # SQLAlchemy models, migrations, queries
│   ├── models/       # MonitoringSession, TrapImage, DetectedObject
│   └── migrations/   # Alembic migrations
├── ml/               # Machine learning pipeline
│   ├── models/       # Detection, classification, tracking models
│   └── pipeline.py   # Orchestrates full processing flow
├── tests/            # Test suite with sample images
├── ui/               # Kivy GUI application
└── settings.py       # Pydantic settings (reads .env, env vars)
```

---

## 🔑 Key Modules

### 1. **trapdata.ml.pipeline** - ML Pipeline Orchestrator
- Entry point for all ML processing
- Coordinates detection → classification → tracking
- Batch processing for efficiency
- **Critical files:**
  - `trapdata/ml/pipeline.py`
  - `trapdata/ml/models/localization.py` - FasterRCNN detection
  - `trapdata/ml/models/classification.py` - Species classifiers
  - `trapdata/ml/models/tracking.py` - Feature extraction & matching

### 2. **trapdata.db** - Database Layer
- SQLAlchemy ORM models
- **Key models:**
  - `MonitoringSession` - A monitoring period (e.g., one night)
  - `TrapImage` - Source image with metadata
  - `DetectedObject` - Detected insect with bbox, classification, tracking info
- **Critical files:**
  - `trapdata/db/models/events.py`
  - `trapdata/db/models/images.py`
  - `trapdata/db/models/detections.py`
  - `trapdata/db/queries.py`

### 3. **trapdata.api** - REST API
- FastAPI application for remote inference
- **Main endpoint:** `POST /process` - Process images through pipeline
- **Critical files:**
  - `trapdata/api/api.py`
  - `trapdata/api/schemas.py` - Pydantic request/response models
  - `trapdata/api/models/` - API-specific model implementations

### 4. **trapdata.cli** - Command Line Interface
- Typer-based CLI with subcommands
- **Critical files:**
  - `trapdata/cli/base.py` - Main entry point
  - `trapdata/cli/queue.py` - Queue management
  - `trapdata/cli/export.py` - Data export

---

## 🛠️ Common Commands

```bash
# Run tests (ALWAYS run before committing)
ami test all
pytest

# Code formatting (run before commit)
black trapdata/
isort trapdata/
flake8 trapdata/

# Database migrations
ami db migrate
alembic revision --autogenerate -m "description"
alembic upgrade head

# Import and process images
ami import --no-queue
ami queue sample --sample-size 10
ami run

# Show data
ami show settings
ami show sessions
ami show occurrences

# Export results
ami export occurrences --format json --collect-images

# Start API server
ami api

# Start Gradio demo
ami gradio
```

---

## ⚠️ Gotchas and Learnings

### Database
- **Queue system:** Images and detections have `in_queue` boolean flag
- **Aggregates are cached:** Call `update_aggregates()` after batch operations
- **Previous frame references:** Used for tracking, can be None for first frame
- **Bulk operations:** Use SQLAlchemy bulk methods for performance

### ML Pipeline
- **Batch processing is critical:** Set appropriate batch sizes in settings
- **Device management:** Models auto-detect GPU availability
- **Model downloads:** First run downloads models to user data directory
- **Image preprocessing:** Models expect RGB format, handle EXIF orientation

### Testing
- **Use test images:** Sample images in `trapdata/tests/images/`
- **Temporary database:** Tests use in-memory SQLite
- **Pipeline tests:** `ami test pipeline` runs end-to-end test

### Configuration
- **Settings priority:** Environment variables > .env file > Kivy config
- **Prefix for env vars:** All env vars use `AMI_` prefix
- **User data paths:** Platform-specific (macOS, Linux, Windows)

### API
- **Image input:** Supports both URLs and base64 encoded images
- **Pipeline selection:** Use slug like "uk_denmark_moths_2023"
- **Response size:** Can be large with many detections, consider pagination

---

## 🔍 Finding Things Quickly

### "Where is X defined?"

```bash
# Find a class definition
rg "^class MonitoringSession"

# Find a function
rg "def process_images"

# Find imports
rg "from trapdata.ml import"

# Find API endpoints
rg "@app\.(get|post)"

# Find database queries
rg "select\(|query\("
```

### "How does X work?"

- **Import process:** See `trapdata/cli/collect.py` and `trapdata/common/filemanagement.py`
- **Queue system:** See `trapdata/db/models/queue.py` and `trapdata/cli/queue.py`
- **ML inference:** See `trapdata/ml/pipeline.py`
- **Tracking:** See `trapdata/ml/models/tracking.py` and tracking fields in `DetectedObject`
- **Export:** See `trapdata/cli/export.py` and `report_data()` methods

---

## 🎯 Development Workflow

1. **Understand the issue** - Read the problem, check existing code
2. **Write tests first** (TDD) - Define expected behavior
3. **Implement minimally** - Simplest solution that works
4. **Run tests** - `ami test all` or `pytest`
5. **Format code** - `black .` and `isort .`
6. **Lint** - `flake8 trapdata/`
7. **Commit** - Small, focused commits
8. **Push** - To feature branch

---

## 📚 Further Documentation

For detailed module documentation, see `docs/claude/` directory:
- `architecture.md` - Detailed architecture and design decisions
- `database.md` - Database schema and query patterns
- `ml-pipeline.md` - ML models and processing flow
- `api.md` - API endpoints and schemas
- `testing.md` - Testing strategies and examples
- `common-tasks.md` - How to accomplish common tasks

---

## 💡 Tips for AI Agents

1. **Use command-line tools** - `rg`, `fd`, `jq`, `git` are cheaper than file reads
2. **Batch similar operations** - Group related changes together
3. **Check existing patterns** - Look for similar code before writing new code
4. **Use type hints** - They help understand function signatures without reading docs
5. **Follow the data flow** - Image → MonitoringSession → TrapImage → DetectedObject → Occurrence
6. **Test incrementally** - Don't wait until everything is done to test
7. **Read test files** - They show how to use the code correctly

---

**Last Updated:** 2025-11-19
**Maintainer:** RolnickLab
**Repository:** https://github.com/RolnickLab/ami-data-companion
