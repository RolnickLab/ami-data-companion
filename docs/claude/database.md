# Database Documentation

## Schema Overview

### Entity-Relationship Diagram

```
┌────────────────────────┐
│  MonitoringSession     │
│  ────────────────────  │
│  id (PK)               │
│  day                   │
│  base_directory        │
│  start_time            │
│  end_time              │
│  num_images (cached)   │
│  num_detected_objects  │
└────────────────────────┘
         │ 1
         │
         │ has many
         │
         ↓ N
┌────────────────────────┐
│     TrapImage          │
│  ────────────────────  │
│  id (PK)               │
│  monitoring_session_id │◄─────┐
│  path                  │      │
│  timestamp             │      │
│  width, height         │      │
│  in_queue              │      │
│  last_processed        │      │
│  num_detected_objects  │      │
└────────────────────────┘      │
         │ 1                    │
         │                      │
         │ has many             │
         │                      │
         ↓ N                    │
┌────────────────────────┐      │
│   DetectedObject       │      │
│  ────────────────────  │      │
│  id (PK)               │      │
│  image_id              │      │
│  monitoring_session_id │──────┘
│  bbox (JSON)           │
│  binary_label          │
│  specific_label        │
│  sequence_id (track)   │◄──┐
│  sequence_frame        │   │ Tracking relationship
│  sequence_previous_id  │───┘ (self-referential)
│  cnn_features (JSON)   │
│  in_queue              │
└────────────────────────┘
```

## Table Details

### MonitoringSession

Represents a single monitoring period, typically one night of camera trap operation.

**Columns:**
- `id` (Integer, PK): Unique identifier
- `day` (Date): Date of monitoring session
- `base_directory` (String): Root path to images
- `start_time` (DateTime): Timestamp of first image
- `end_time` (DateTime): Timestamp of last image
- `notes` (JSON): Flexible metadata storage
- `num_images` (Integer): Cached count of images
- `num_detected_objects` (Integer): Cached count of detections

**Indexes:**
- Primary key on `id`
- Index on `day` for date range queries

**Common Queries:**
```python
# Get session by date
session = db.query(MonitoringSession).filter_by(day=target_date).first()

# Get sessions in date range
sessions = db.query(MonitoringSession).filter(
    MonitoringSession.day.between(start_date, end_date)
).all()

# Get session with image count
session = db.query(MonitoringSession).options(
    selectinload(MonitoringSession.images)
).first()
```

**Business Methods:**
- `duration()`: Calculate time span of session
- `update_aggregates()`: Refresh cached counts
- `report_data()`: Export format for JSON/CSV

### TrapImage

Represents a source image from the camera trap.

**Columns:**
- `id` (Integer, PK): Unique identifier
- `monitoring_session_id` (Integer, FK): Parent session
- `base_path` (String): Base directory path
- `path` (String): Relative path to image file
- `timestamp` (DateTime): Image capture time
- `filesize` (Integer): File size in bytes
- `width` (Integer): Image width in pixels
- `height` (Integer): Image height in pixels
- `last_read` (DateTime): Last time file was accessed
- `last_processed` (DateTime): Last time processed through pipeline
- `in_queue` (Boolean): Currently in processing queue
- `notes` (JSON): Flexible metadata
- `num_detected_objects` (Integer): Cached detection count

**Indexes:**
- Primary key on `id`
- Foreign key on `monitoring_session_id`
- Index on `timestamp` for temporal queries
- Index on `in_queue` for queue management

**Common Queries:**
```python
# Get images in queue
images = db.query(TrapImage).filter_by(in_queue=True).limit(100).all()

# Get unprocessed images
images = db.query(TrapImage).filter(
    TrapImage.last_processed.is_(None)
).all()

# Get images with detections
images = db.query(TrapImage).filter(
    TrapImage.num_detected_objects > 0
).all()

# Get images in time range
images = db.query(TrapImage).filter(
    TrapImage.timestamp.between(start_time, end_time)
).order_by(TrapImage.timestamp).all()
```

**Business Methods:**
- `previous_image()`: Get chronologically previous image
- `next_image()`: Get chronologically next image
- `update_source_data()`: Refresh EXIF metadata
- `full_path()`: Construct absolute file path
- `report_data()`: Export format

**Queue Management:**
```python
# Add to queue
image.in_queue = True
db.commit()

# Remove from queue
image.in_queue = False
image.last_processed = datetime.now()
db.commit()
```

### DetectedObject

Represents a single detected insect in an image.

**Columns:**

*Identity:*
- `id` (Integer, PK): Unique identifier
- `image_id` (Integer, FK): Parent image
- `monitoring_session_id` (Integer, FK): Parent session

*Bounding Box:*
- `bbox` (JSON): Bounding box `[x1, y1, x2, y2]` in pixels
- `area_pixels` (Integer): Bounding box area
- `source_image_width` (Integer): Width of source image
- `source_image_height` (Integer): Height of source image

*Classification:*
- `binary_label` (String): "moth" or "nonmoth"
- `binary_label_score` (Float): Binary confidence [0-1]
- `specific_label` (String): Species name
- `specific_label_score` (Float): Species confidence [0-1]
- `model_name` (String): Model used for detection

*Tracking:*
- `sequence_id` (String): Unique identifier for track
- `sequence_frame` (Integer): Frame number in track
- `sequence_previous_id` (Integer, FK): Previous detection in track
- `sequence_previous_cost` (Float): Matching cost for tracking
- `cnn_features` (JSON): Feature vector for matching

*Metadata:*
- `path` (String): Path to cropped image
- `timestamp` (DateTime): Detection timestamp
- `last_detected` (DateTime): When detection was created
- `in_queue` (Boolean): In processing queue
- `notes` (JSON): Flexible metadata

**Indexes:**
- Primary key on `id`
- Foreign keys on `image_id`, `monitoring_session_id`, `sequence_previous_id`
- Index on `sequence_id` for track queries
- Index on `specific_label` for species queries
- Index on `in_queue` for queue management

**Common Queries:**
```python
# Get detections for an image
detections = db.query(DetectedObject).filter_by(image_id=img.id).all()

# Get all detections of a species
moths = db.query(DetectedObject).filter_by(
    specific_label="Actias luna"
).all()

# Get detections in a track
track = db.query(DetectedObject).filter_by(
    sequence_id="track_123"
).order_by(DetectedObject.sequence_frame).all()

# Get high-confidence detections
confident = db.query(DetectedObject).filter(
    DetectedObject.specific_label_score > 0.9
).all()

# Get detections needing classification
unclassified = db.query(DetectedObject).filter(
    DetectedObject.specific_label.is_(None)
).all()
```

**Business Methods:**
- `cropped_image_data()`: Get PIL Image of detection
- `save_cropped_image_data()`: Save crop to disk
- `previous_frame_detections()`: Get detections from previous frame
- `best_sibling()`: Get highest-confidence detection in same track
- `track_info()`: Statistics about the track
- `report_data()`: Export format

**Tracking Logic:**
```python
# Create new track
detection.sequence_id = f"track_{uuid.uuid4()}"
detection.sequence_frame = 0

# Link to previous detection
detection.sequence_id = previous_detection.sequence_id
detection.sequence_frame = previous_detection.sequence_frame + 1
detection.sequence_previous_id = previous_detection.id
detection.sequence_previous_cost = matching_cost
```

## Occurrence Concept

**Occurrences** represent tracked individuals across multiple frames. They are not stored in the database but computed from `sequence_id` groupings.

**Pydantic Model:**
```python
class Occurrence:
    id: str  # sequence_id
    label: str  # species name
    best_score: float  # highest confidence
    start_time: datetime
    end_time: datetime
    duration: timedelta
    num_frames: int
    examples: List[DetectedObject]  # Sample detections
```

**Computing Occurrences:**
```python
# Group detections by sequence_id
occurrences = db.query(
    DetectedObject.sequence_id,
    func.min(DetectedObject.timestamp).label('start_time'),
    func.max(DetectedObject.timestamp).label('end_time'),
    func.count(DetectedObject.id).label('num_frames'),
    func.max(DetectedObject.specific_label_score).label('best_score')
).group_by(DetectedObject.sequence_id).all()
```

## Queue System

### Queue Tables (Legacy/Optional)

The queue system uses boolean flags on existing tables rather than separate queue tables:

**TrapImage.in_queue:**
- `True`: Image needs processing
- `False`: Image processed or not queued

**DetectedObject.in_queue:**
- `True`: Detection needs classification/tracking
- `False`: Detection fully processed

### Queue Manager

```python
from trapdata.db.models.queue import QueueManager

# Initialize
queue_mgr = QueueManager(db_session)

# Add images to queue
queue_mgr.add_images_to_queue(
    monitoring_session_id=123,
    sample_size=100
)

# Get queue status
status = queue_mgr.get_queue_status()
# Returns: {
#   'source_images': {'queued': 100, 'unprocessed': 500, 'done': 200},
#   'detected_objects': {...},
#   ...
# }

# Get batch for processing
batch = queue_mgr.get_image_batch(batch_size=32)

# Mark complete
for image in batch:
    image.in_queue = False
    image.last_processed = datetime.now()
db_session.commit()
```

## Migration Management

### Alembic Setup

**Configuration:** `alembic.ini` and `trapdata/db/migrations/env.py`

**Current Migrations:**
1. `1678330000_3665528a445c_init_existing_tables.py` - Initial schema
2. `1678332222_90d5f6ae09ec_add_fields_for_tracking.py` - Add tracking fields
3. `1678338134_1544478c3031_missing_data_for_tracking.py` - Complete tracking schema

### Creating Migrations

```bash
# Auto-generate migration
alembic revision --autogenerate -m "Add new field"

# Review generated migration
vim trapdata/db/migrations/versions/XXXX_add_new_field.py

# Apply migration
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Migration Best Practices

1. **Review auto-generated code** - Alembic isn't perfect
2. **Test migrations on copy of production data**
3. **Include data migrations if needed**
4. **Never edit applied migrations** - Create new ones
5. **Keep migrations small and focused**

## Database Configuration

### SQLite (Development/Desktop)

```bash
# .env file
AMI_DATABASE_URL=sqlite:///~/ami-data.db
```

**Benefits:**
- No server required
- Simple setup
- Good for single-user desktop app

**Limitations:**
- No concurrent writes
- Limited performance for large datasets
- No network access

### PostgreSQL (Production/API)

```bash
# .env file
AMI_DATABASE_URL=postgresql://user:pass@localhost/ami_data
```

**Benefits:**
- Concurrent access
- Better performance
- Full SQL features
- Remote access

**Setup:**
```bash
# Start PostgreSQL container
./scripts/start_db_container.sh

# Create database
createdb ami_data

# Run migrations
alembic upgrade head
```

## Query Optimization

### N+1 Query Problem

**Problem:**
```python
# This generates N+1 queries (bad!)
images = db.query(TrapImage).all()
for img in images:
    print(img.monitoring_session.day)  # Separate query per image!
```

**Solution:**
```python
# Use eager loading (good!)
from sqlalchemy.orm import joinedload

images = db.query(TrapImage).options(
    joinedload(TrapImage.monitoring_session)
).all()
for img in images:
    print(img.monitoring_session.day)  # No extra queries!
```

### Bulk Operations

**Problem:**
```python
# Slow: One transaction per detection
for detection_data in detections:
    detection = DetectedObject(**detection_data)
    db.add(detection)
    db.commit()  # Expensive!
```

**Solution:**
```python
# Fast: Bulk insert
db.bulk_insert_mappings(DetectedObject, detections)
db.commit()  # One transaction
```

### Pagination

```python
# Don't load all results at once
page_size = 100
offset = 0

while True:
    batch = db.query(TrapImage).limit(page_size).offset(offset).all()
    if not batch:
        break
    process_batch(batch)
    offset += page_size
```

## Common Patterns

### Find or Create

```python
def find_or_create_session(db, day, base_directory):
    session = db.query(MonitoringSession).filter_by(
        day=day,
        base_directory=base_directory
    ).first()

    if not session:
        session = MonitoringSession(
            day=day,
            base_directory=base_directory
        )
        db.add(session)
        db.commit()

    return session
```

### Update Aggregates

```python
def update_session_aggregates(db, session_id):
    """Update cached counts for a session"""
    session = db.query(MonitoringSession).get(session_id)

    # Count images
    session.num_images = db.query(TrapImage).filter_by(
        monitoring_session_id=session_id
    ).count()

    # Count detections
    session.num_detected_objects = db.query(DetectedObject).filter_by(
        monitoring_session_id=session_id
    ).count()

    db.commit()
```

### Export Data

```python
def export_occurrences(db, format='json'):
    """Export occurrence data"""
    occurrences = db.query(
        DetectedObject.sequence_id,
        DetectedObject.specific_label,
        func.min(DetectedObject.timestamp).label('start_time'),
        func.max(DetectedObject.timestamp).label('end_time'),
        func.count(DetectedObject.id).label('num_frames'),
        func.max(DetectedObject.specific_label_score).label('best_score')
    ).filter(
        DetectedObject.sequence_id.isnot(None)
    ).group_by(
        DetectedObject.sequence_id,
        DetectedObject.specific_label
    ).all()

    if format == 'json':
        return [occ._asdict() for occ in occurrences]
    elif format == 'csv':
        return pd.DataFrame(occurrences).to_csv()
```

## Testing Database Code

### Using In-Memory Database

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from trapdata.db.base import Base

@pytest.fixture
def db_session():
    """Create in-memory database for testing"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()

def test_create_monitoring_session(db_session):
    session = MonitoringSession(
        day=date(2024, 1, 1),
        base_directory="/tmp/test"
    )
    db_session.add(session)
    db_session.commit()

    assert session.id is not None
    assert db_session.query(MonitoringSession).count() == 1
```

## Troubleshooting

### Common Issues

**1. Migration conflicts:**
```bash
# Reset migrations (DESTRUCTIVE!)
alembic stamp head
```

**2. Foreign key violations:**
```python
# Always create parent before child
session = MonitoringSession(...)
db.add(session)
db.flush()  # Get session.id

image = TrapImage(monitoring_session_id=session.id, ...)
db.add(image)
db.commit()
```

**3. Stale data:**
```python
# Refresh object from database
db.refresh(image)

# Or expire all and reload
db.expire_all()
```

**4. Transaction deadlocks:**
```python
# Keep transactions short
try:
    # Do work
    db.commit()
except Exception:
    db.rollback()
    raise
```
