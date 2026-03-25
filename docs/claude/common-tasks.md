# Common Tasks Guide

This guide shows how to accomplish common development and operational tasks.

## Development Tasks

### Adding a New Species Classification Model

**1. Create model class:**

```python
# trapdata/ml/models/classification.py

from trapdata.ml.models.base import SpeciesClassifier

class MothClassifierMyRegion(SpeciesClassifier):
    slug = "my_region_moths_2025"
    name = "My Region Moth Classifier"
    version = "1.0"
    model_url = "https://example.com/models/my_region_2025.pth"

    category_map = {
        0: "Species name 1",
        1: "Species name 2",
        2: "Species name 3",
        # ... add all species
    }

    def __init__(self):
        super().__init__()
        # Model will be auto-downloaded from model_url
        # and loaded via load_model() method
```

**2. Model is automatically registered:**
```python
# No manual registration needed!
# Import triggers auto-registration
from trapdata.ml.models.classification import MothClassifierMyRegion
```

**3. Use the model:**
```bash
# Via CLI
AMI_SPECIES_CLASSIFICATION_MODEL=my_region_moths_2025 ami run

# Via API
curl -X POST http://localhost:2000/process \
  -d '{"pipeline": "my_region_moths_2025", ...}'
```

### Adding a Database Field

**1. Update the model:**
```python
# trapdata/db/models/detections.py

class DetectedObject(Base):
    # ... existing fields ...

    # Add new field
    my_new_field = Column(String, nullable=True)
```

**2. Create migration:**
```bash
alembic revision --autogenerate -m "Add my_new_field to detections"
```

**3. Review generated migration:**
```python
# trapdata/db/migrations/versions/XXXX_add_my_new_field.py

def upgrade():
    op.add_column('detected_objects',
        sa.Column('my_new_field', sa.String(), nullable=True))

def downgrade():
    op.drop_column('detected_objects', 'my_new_field')
```

**4. Apply migration:**
```bash
alembic upgrade head
```

### Adding a CLI Command

**1. Add command to CLI:**
```python
# trapdata/cli/base.py

@cli.command()
def my_command(
    option: str = typer.Option(..., help="Description"),
    flag: bool = typer.Option(False, help="Enable flag")
):
    """
    Description of my command.

    This is shown in --help output.
    """
    from trapdata.db.base import get_session

    with get_session() as db:
        # Do work
        if flag:
            print(f"Processing with option: {option}")
```

**2. Use the command:**
```bash
ami my-command --option value --flag
```

### Adding an API Endpoint

**1. Add endpoint:**
```python
# trapdata/api/api.py

from trapdata.api.schemas import MyRequest, MyResponse

@app.post("/my-endpoint", response_model=MyResponse)
async def my_endpoint(request: MyRequest):
    """
    Process my custom request.
    """
    # Do work
    result = process_request(request)
    return MyResponse(**result)
```

**2. Define schemas:**
```python
# trapdata/api/schemas.py

class MyRequest(BaseModel):
    field1: str
    field2: int

class MyResponse(BaseModel):
    result: str
    count: int
```

**3. Test endpoint:**
```bash
curl -X POST http://localhost:2000/my-endpoint \
  -H "Content-Type: application/json" \
  -d '{"field1": "value", "field2": 42}'
```

### Adding Tests

**1. Create test file:**
```python
# trapdata/tests/test_my_feature.py

import pytest
from trapdata.db.base import get_session
from trapdata.db.models.detections import DetectedObject

@pytest.fixture
def db_session():
    """Create in-memory database"""
    # Setup
    from sqlalchemy import create_engine
    from trapdata.db.base import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()

def test_my_feature(db_session):
    """Test my feature works correctly"""
    # Arrange
    detection = DetectedObject(
        bbox=[0, 0, 100, 100],
        specific_label="Test species"
    )
    db_session.add(detection)
    db_session.commit()

    # Act
    result = db_session.query(DetectedObject).first()

    # Assert
    assert result.specific_label == "Test species"
    assert result.bbox == [0, 0, 100, 100]
```

**2. Run tests:**
```bash
pytest trapdata/tests/test_my_feature.py -v
```

## Operational Tasks

### Import Images from a Deployment

**1. Organize images:**
```bash
# Images should be in a directory structure like:
# /path/to/deployment/
#   2024-01-01/
#     image001.jpg
#     image002.jpg
#   2024-01-02/
#     image001.jpg
```

**2. Set base path:**
```bash
export AMI_IMAGE_BASE_PATH=/path/to/deployment
```

**3. Import:**
```bash
ami import --no-queue
```

**4. Verify:**
```bash
ami show sessions
```

### Process a Sample of Images

**1. Import images:**
```bash
ami import --no-queue
```

**2. Queue sample:**
```bash
ami queue sample --sample-size 100
```

**3. Check queue:**
```bash
ami queue status
```

**4. Process:**
```bash
ami run
```

**5. View results:**
```bash
ami show occurrences
```

### Export Results

**Export occurrences to JSON:**
```bash
ami export occurrences --format json --output results.json
```

**Export with cropped images:**
```bash
ami export occurrences --format json --collect-images --output results.json
# Creates results.json and results_images/ directory
```

**Export to CSV:**
```bash
ami export occurrences --format csv --output results.csv
```

**Export specific session:**
```bash
ami export occurrences --session-id 123 --format json
```

### Reprocess with Different Model

**1. Update configuration:**
```bash
export AMI_SPECIES_CLASSIFICATION_MODEL=global_moths_2024
```

**2. Queue already-processed images:**
```bash
ami queue all --force
```

**3. Process:**
```bash
ami run
```

### Fix Tracking After Parameter Change

**1. Clear existing tracks:**
```python
from trapdata.db.base import get_session
from trapdata.db.models.detections import DetectedObject

with get_session() as db:
    db.query(DetectedObject).update({
        "sequence_id": None,
        "sequence_frame": None,
        "sequence_previous_id": None,
        "sequence_previous_cost": None
    })
    db.commit()
```

**2. Adjust tracking threshold:**
```bash
# In settings or environment
export AMI_TRACKING_COST_THRESHOLD=0.8
```

**3. Rerun tracking:**
```bash
ami run  # Will track detections that have features but no sequence_id
```

### Backup Database

**SQLite:**
```bash
# Simple copy
cp ~/ami-data.db ~/ami-data-backup-$(date +%Y%m%d).db

# Or use SQLite backup command
sqlite3 ~/ami-data.db ".backup ~/ami-data-backup.db"
```

**PostgreSQL:**
```bash
pg_dump ami_data > ami_data_backup_$(date +%Y%m%d).sql

# Restore
psql ami_data < ami_data_backup_20240115.sql
```

### Migrate Database to PostgreSQL

**1. Export from SQLite:**
```bash
ami export all --format json --output export.json
```

**2. Setup PostgreSQL:**
```bash
createdb ami_data
export AMI_DATABASE_URL=postgresql://user:pass@localhost/ami_data
```

**3. Run migrations:**
```bash
alembic upgrade head
```

**4. Import data:**
```bash
ami import-data export.json
```

### Clear Queue

**Clear all queues:**
```bash
ami queue clear
```

**Clear specific queue:**
```python
from trapdata.db.base import get_session
from trapdata.db.models.images import TrapImage

with get_session() as db:
    db.query(TrapImage).update({"in_queue": False})
    db.commit()
```

### View Processing Progress

**Check queue status:**
```bash
ami queue status
```

**Monitor in real-time:**
```bash
watch -n 5 'ami queue status'
```

**Check logs:**
```bash
# Logs are written to stdout
ami run 2>&1 | tee processing.log
```

## Debugging Tasks

### Find Images Not Processing

**Images in queue but not processing:**
```python
from trapdata.db.base import get_session
from trapdata.db.models.images import TrapImage

with get_session() as db:
    stuck = db.query(TrapImage).filter(
        TrapImage.in_queue == True,
        TrapImage.last_processed.isnot(None)
    ).all()

    for img in stuck:
        print(f"Image {img.id}: {img.path}")
        print(f"  Last processed: {img.last_processed}")
        print(f"  Detections: {img.num_detected_objects}")
```

### Check Detection Quality

**Find low-confidence detections:**
```python
from trapdata.db.base import get_session
from trapdata.db.models.detections import DetectedObject

with get_session() as db:
    low_conf = db.query(DetectedObject).filter(
        DetectedObject.specific_label_score < 0.5,
        DetectedObject.specific_label.isnot(None)
    ).all()

    for det in low_conf:
        print(f"{det.specific_label}: {det.specific_label_score:.2%}")
```

### Inspect Track Quality

**Find short tracks:**
```python
from trapdata.db.base import get_session
from trapdata.db.models.detections import DetectedObject
from sqlalchemy import func

with get_session() as db:
    tracks = db.query(
        DetectedObject.sequence_id,
        func.count(DetectedObject.id).label('count')
    ).filter(
        DetectedObject.sequence_id.isnot(None)
    ).group_by(
        DetectedObject.sequence_id
    ).having(
        func.count(DetectedObject.id) == 1
    ).all()

    print(f"Found {len(tracks)} single-frame tracks")
```

**Find broken tracks:**
```python
# Tracks with gaps in sequence_frame numbers
from trapdata.db.base import get_session
from trapdata.db.models.detections import DetectedObject

with get_session() as db:
    for sequence_id in get_all_sequence_ids(db):
        detections = db.query(DetectedObject).filter_by(
            sequence_id=sequence_id
        ).order_by(DetectedObject.sequence_frame).all()

        frames = [d.sequence_frame for d in detections]
        expected = list(range(len(detections)))

        if frames != expected:
            print(f"Track {sequence_id} has gaps: {frames}")
```

### Debug Model Loading

**Test model loads correctly:**
```python
from trapdata.ml.models.classification import SpeciesClassifier

# Load model
clf = SpeciesClassifier.get_model("uk_denmark_moths_2023")()

# Check model is loaded
assert clf.model is not None
print(f"Model loaded: {clf.name} v{clf.version}")
print(f"Categories: {len(clf.category_map)}")

# Test prediction
from PIL import Image
import torch

test_image = Image.new('RGB', (224, 224))
result = clf.predict([test_image])
print(f"Prediction: {result}")
```

### Check Database Integrity

**Count records:**
```python
from trapdata.db.base import get_session
from trapdata.db.models.events import MonitoringSession
from trapdata.db.models.images import TrapImage
from trapdata.db.models.detections import DetectedObject

with get_session() as db:
    sessions = db.query(MonitoringSession).count()
    images = db.query(TrapImage).count()
    detections = db.query(DetectedObject).count()

    print(f"Sessions: {sessions}")
    print(f"Images: {images}")
    print(f"Detections: {detections}")
    print(f"Avg detections/image: {detections/images if images else 0:.2f}")
```

**Find orphaned records:**
```python
# Detections without parent image
orphaned = db.query(DetectedObject).filter(
    ~DetectedObject.image_id.in_(
        db.query(TrapImage.id)
    )
).count()

print(f"Orphaned detections: {orphaned}")
```

### Performance Profiling

**Profile pipeline:**
```python
import cProfile
import pstats

# Profile code
profiler = cProfile.Profile()
profiler.enable()

# Run pipeline
run_pipeline(db, settings)

profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

**Check GPU usage:**
```bash
# Monitor GPU while processing
watch -n 1 nvidia-smi

# Check if PyTorch sees GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Maintenance Tasks

### Update Cached Counts

**Update all monitoring session aggregates:**
```python
from trapdata.db.base import get_session
from trapdata.db.models.events import MonitoringSession

with get_session() as db:
    sessions = db.query(MonitoringSession).all()
    for session in sessions:
        session.update_aggregates()
    db.commit()
```

### Clean Up Old Data

**Delete images from specific date:**
```python
from trapdata.db.base import get_session
from trapdata.db.models.images import TrapImage
from datetime import date

with get_session() as db:
    db.query(TrapImage).filter(
        TrapImage.timestamp < date(2024, 1, 1)
    ).delete()
    db.commit()
```

**Delete low-confidence detections:**
```python
from trapdata.db.base import get_session
from trapdata.db.models.detections import DetectedObject

with get_session() as db:
    db.query(DetectedObject).filter(
        DetectedObject.specific_label_score < 0.1
    ).delete()
    db.commit()
```

### Rebuild Tracking

**Delete all tracking data:**
```python
from trapdata.db.base import get_session
from trapdata.db.models.detections import DetectedObject

with get_session() as db:
    db.query(DetectedObject).update({
        "sequence_id": None,
        "sequence_frame": None,
        "sequence_previous_id": None,
        "sequence_previous_cost": None
    })
    db.commit()
```

**Rerun tracking:**
```bash
ami run  # Will only run tracking stage
```

### Update All Images' EXIF Data

```python
from trapdata.db.base import get_session
from trapdata.db.models.images import TrapImage

with get_session() as db:
    images = db.query(TrapImage).all()
    for img in images:
        img.update_source_data()
    db.commit()
```

## Integration Tasks

### Integrate with External Database

**Export to external format:**
```python
import pandas as pd
from trapdata.db.base import get_session
from trapdata.db.models.detections import DetectedObject

with get_session() as db:
    detections = db.query(DetectedObject).all()

    # Convert to DataFrame
    df = pd.DataFrame([d.report_data() for d in detections])

    # Export to various formats
    df.to_csv('detections.csv', index=False)
    df.to_excel('detections.xlsx', index=False)
    df.to_parquet('detections.parquet', index=False)

    # Or to database
    import sqlalchemy
    engine = sqlalchemy.create_engine('postgresql://...')
    df.to_sql('detections', engine, if_exists='replace')
```

### Sync with Cloud Storage

**Upload images to S3:**
```python
import boto3
from trapdata.db.base import get_session
from trapdata.db.models.images import TrapImage

s3 = boto3.client('s3')
bucket = 'my-ami-data'

with get_session() as db:
    images = db.query(TrapImage).all()
    for img in images:
        s3.upload_file(
            img.full_path(),
            bucket,
            f"images/{img.monitoring_session_id}/{img.path}"
        )
```

### Generate Report

**Create HTML report:**
```python
from trapdata.db.base import get_session
from trapdata.db.models.events import MonitoringSession
from jinja2 import Template

template = Template("""
<html>
<head><title>AMI Report</title></head>
<body>
  <h1>Monitoring Sessions</h1>
  {% for session in sessions %}
  <div>
    <h2>{{ session.day }}</h2>
    <p>Images: {{ session.num_images }}</p>
    <p>Detections: {{ session.num_detected_objects }}</p>
    <p>Duration: {{ session.duration() }}</p>
  </div>
  {% endfor %}
</body>
</html>
""")

with get_session() as db:
    sessions = db.query(MonitoringSession).all()
    html = template.render(sessions=sessions)

    with open('report.html', 'w') as f:
        f.write(html)
```
