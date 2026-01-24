# Architecture Documentation

## System Architecture

### High-Level Overview

AMI Data Companion follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │  CLI (Typer) │  │  GUI (Kivy)  │  │  API (FastAPI)     │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Business Logic Layer                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ML Pipeline (trapdata.ml)                   │   │
│  │  • Localization (Object Detection)                       │   │
│  │  • Binary Classification (moth/non-moth)                 │   │
│  │  • Species Classification                                │   │
│  │  • Feature Extraction                                    │   │
│  │  • Tracking (Cross-frame matching)                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Data Access Layer                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Database Layer (trapdata.db)                    │  │
│  │  • SQLAlchemy ORM Models                                  │  │
│  │  • Query Builders                                         │  │
│  │  • Migration Management (Alembic)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Storage Layer                           │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │   Database   │  │  File System   │  │  Model Weights   │   │
│  │ (SQLite/PG)  │  │  (Images)      │  │  (User Data)     │   │
│  └──────────────┘  └────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Design Patterns

### 1. Pipeline Pattern (ML Processing)

The ML pipeline follows a sequential processing pattern with well-defined stages:

```python
# trapdata/ml/pipeline.py
Image → Localization → Binary Classification → Species Classification → Features → Tracking
```

**Benefits:**
- Each stage can be developed/tested independently
- Easy to add/remove stages
- Natural queue-based processing
- Supports batch processing at each stage

**Implementation:**
- `trapdata/ml/pipeline.py` - Orchestrates the pipeline
- Each stage reads from database, processes, writes back
- Queue flags (`in_queue`) track processing status

### 2. Repository Pattern (Database Access)

Database models encapsulate both data and behavior:

```python
# Example: DetectedObject model
class DetectedObject(Base):
    # Data fields
    bbox = Column(JSON)
    specific_label = Column(String)

    # Business logic methods
    def previous_frame_detections(self):
        """Get detections from previous frame for tracking"""
        ...

    def best_sibling(self):
        """Get best detection in same track"""
        ...
```

**Benefits:**
- Business logic close to data
- Consistent query patterns
- Easy to test
- Encapsulation of complex queries

### 3. Registry Pattern (Model Management)

ML models self-register via inheritance:

```python
# Base class with registration
class SpeciesClassifier(InferenceBaseClass):
    _registry = {}

    def __init_subclass__(cls):
        # Auto-register subclasses
        cls._registry[cls.slug] = cls

# Automatic discovery
MothClassifierPanama()  # Auto-registers as "panama_moths_2023"
```

**Benefits:**
- No manual registration needed
- Easy to add new models
- Dynamic model discovery
- Validates model metadata

### 4. Settings Pattern (Configuration)

Pydantic-based settings with multiple sources:

```python
# trapdata/settings.py
class Settings(BaseSettings):
    database_url: str
    localization_model: str

    class Config:
        env_prefix = "AMI_"
        env_file = ".env"
```

**Priority:** Environment variables > .env file > Kivy config > defaults

**Benefits:**
- Type-safe configuration
- Validation at startup
- Multiple configuration sources
- Easy testing with overrides

## Module Dependencies

### Dependency Graph

```
trapdata.ui ────────┐
                    ↓
trapdata.cli ───→ trapdata.ml ───→ trapdata.db ───→ trapdata.common
                    ↓                  ↓
                trapdata.settings      ↓
                                       ↓
trapdata.api ───────────────────────→ trapdata.db
```

**Rules:**
- No circular dependencies
- Common utilities at bottom (trapdata.common)
- Settings accessible from all layers
- UI/CLI/API are parallel entry points

## Data Architecture

### Entity Relationships

```
MonitoringSession (1) ──────< (many) TrapImage
       │                         │
       │                         │
       └──────< (many) ──────────┘
                  │
                  ↓
           DetectedObject
                  │
                  │ (tracking)
                  ↓
            sequence_id (groups into Occurrences)
```

### Data Flow States

**TrapImage states:**
1. `discovered` - Found in filesystem
2. `in_queue=True` - Queued for processing
3. `last_processed!=None` - Processed
4. Has detections → DetectedObject records created

**DetectedObject states:**
1. `created` - Bounding box detected
2. `binary_label set` - Binary classification done
3. `specific_label set` - Species classification done
4. `cnn_features set` - Features extracted
5. `sequence_id set` - Tracked across frames

## Concurrency Model

### Queue-Based Processing

The system uses a queue-based model for concurrent processing:

```python
# Queue management
queue_manager = QueueManager(session)
queue_manager.add_images_to_queue(sample_size=100)

# Process in batches
while True:
    batch = queue_manager.get_batch(batch_size=32)
    if not batch:
        break
    process_batch(batch)
    queue_manager.mark_complete(batch)
```

**Benefits:**
- Supports distributed processing
- Graceful failure handling
- Can pause/resume
- Progress tracking

### Threading Considerations

- **ML Inference:** Batch processing on GPU (no threading within model)
- **Database:** SQLAlchemy session per thread
- **File I/O:** Can parallelize image loading
- **API:** FastAPI handles concurrent requests via async

## Extension Points

### 1. Adding New ML Models

Inherit from base class and implement required methods:

```python
from trapdata.ml.models.classification import SpeciesClassifier

class MyNewClassifier(SpeciesClassifier):
    slug = "my_classifier_2025"
    name = "My Classifier"
    version = "1.0"

    def __init__(self):
        # Load your model
        self.model = load_model()

    def predict(self, images):
        # Implement prediction
        return self.model(images)
```

### 2. Adding New API Endpoints

Add to FastAPI app:

```python
# trapdata/api/api.py
@app.post("/my-endpoint")
async def my_endpoint(request: MyRequest) -> MyResponse:
    # Implementation
    pass
```

### 3. Adding New CLI Commands

Add to Typer CLI:

```python
# trapdata/cli/base.py
@cli.command()
def my_command(option: str = typer.Option(...)):
    """My command description"""
    # Implementation
    pass
```

### 4. Adding Database Fields

Create Alembic migration:

```bash
alembic revision --autogenerate -m "Add new field"
alembic upgrade head
```

## Performance Considerations

### Database Optimization

1. **Bulk Operations:**
   ```python
   # Bad: One query per item
   for item in items:
       session.add(item)
       session.commit()

   # Good: Bulk insert
   session.bulk_insert_mappings(DetectedObject, items)
   session.commit()
   ```

2. **Eager Loading:**
   ```python
   # Bad: N+1 queries
   images = session.query(TrapImage).all()
   for img in images:
       print(img.monitoring_session.day)  # Query per image

   # Good: Join/eager load
   images = session.query(TrapImage).options(
       joinedload(TrapImage.monitoring_session)
   ).all()
   ```

3. **Indexing:**
   - Foreign keys are indexed
   - Add indexes for common queries (e.g., timestamp ranges)

### ML Pipeline Optimization

1. **Batch Processing:**
   - Use largest batch size that fits in GPU memory
   - Typical: 32-128 images per batch

2. **Prefetching:**
   - Load next batch while processing current
   - PyTorch DataLoader handles this automatically

3. **Model Caching:**
   - Models loaded once and kept in memory
   - Weights downloaded to user data directory

### File I/O Optimization

1. **Lazy Loading:**
   - Only load image data when needed
   - Store paths, not image data

2. **Caching:**
   - Cache image metadata (size, EXIF)
   - Avoid repeated file reads

## Testing Architecture

### Test Pyramid

```
        ┌────────────┐
        │  E2E Tests │  (Few) - Full pipeline tests
        └────────────┘
       ┌──────────────┐
       │ Integration  │   (Some) - API, DB, ML together
       └──────────────┘
      ┌────────────────┐
      │  Unit Tests    │    (Many) - Individual functions
      └────────────────┘
```

**Test Organization:**
- Unit tests: Close to code (e.g., `trapdata/db/tests.py`)
- Integration tests: `trapdata/tests/` directory
- E2E tests: `trapdata/tests/test_pipeline.py`

### Test Database

Tests use separate database (in-memory SQLite):

```python
# Test setup
settings.database_url = "sqlite:///:memory:"
Base.metadata.create_all(engine)
```

## Security Considerations

1. **SQL Injection:** Prevented by SQLAlchemy ORM
2. **Path Traversal:** Validated in file operations
3. **API Authentication:** Not implemented (local use)
4. **Model Security:** Models from trusted sources only
5. **Error Handling:** Sensitive info not in error messages

## Monitoring and Logging

### Structured Logging

```python
import structlog
logger = structlog.get_logger()

logger.info("Processing image",
    image_id=img.id,
    detections=len(results))
```

**Benefits:**
- Machine-parseable logs
- Contextual information
- Easy filtering and aggregation

### Error Tracking

Sentry integration for production:

```python
import sentry_sdk
sentry_sdk.init(dsn=settings.sentry_dsn)
```

## Future Architecture Considerations

1. **Distributed Processing:**
   - Queue-based design supports remote workers
   - Could use Celery or similar

2. **Cloud Storage:**
   - S3 support already present (boto3)
   - Could store images in cloud

3. **API Scaling:**
   - FastAPI supports async
   - Could add load balancing

4. **Real-time Processing:**
   - WebSocket support for live updates
   - Could process images as captured

5. **Multi-tenancy:**
   - Database supports multiple deployments
   - Could add user accounts
