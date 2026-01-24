# Testing Documentation

## Testing Strategy

The AMI Data Companion uses a multi-layered testing approach:

```
┌─────────────────────────────────────┐
│     End-to-End Tests                │  Full pipeline with real models
│  - test_pipeline.py                 │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│     Integration Tests               │  Multiple components together
│  - API tests                        │
│  - Database + ML tests              │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│     Unit Tests                      │  Individual functions/classes
│  - Model registry tests             │
│  - EXIF parsing tests               │
│  - Utility function tests           │
└─────────────────────────────────────┘
```

## Test Organization

### Test Locations

```
trapdata/
├── tests/                    # Integration & E2E tests
│   ├── images/              # Test images (various deployments)
│   ├── config/              # Test configurations
│   ├── test_pipeline.py     # End-to-end pipeline test
│   ├── test_registry.py     # Model registry tests
│   └── test_exif.py         # EXIF metadata tests
├── api/tests/               # API-specific tests
│   ├── test_api.py          # API endpoint tests
│   └── test_models.py       # API model tests
└── db/tests.py              # Database unit tests
```

### Test Data

**Sample Images:**
- `trapdata/tests/images/cyprus/` - Cyprus deployment
- `trapdata/tests/images/denmark/` - Denmark deployment
- `trapdata/tests/images/panama/` - Panama deployment
- `trapdata/tests/images/vermont/` - Vermont deployment
- `trapdata/tests/images/global/` - Multi-region test set
- `trapdata/tests/images/sequential/` - Sequential frames for tracking

## Running Tests

### Run All Tests

```bash
# Using pytest
pytest

# Using CLI command
ami test all

# With coverage
pytest --cov=trapdata --cov-report=html
```

### Run Specific Tests

```bash
# Single test file
pytest trapdata/tests/test_pipeline.py

# Single test function
pytest trapdata/tests/test_pipeline.py::test_full_pipeline

# Tests matching pattern
pytest -k "test_classification"

# API tests only
pytest trapdata/api/tests/
```

### Run with Different Verbosity

```bash
# Verbose output
pytest -v

# Very verbose (show print statements)
pytest -vv -s

# Quiet (only failures)
pytest -q
```

### Run Tests in Parallel

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run with 4 workers
pytest -n 4
```

## End-to-End Tests

### Full Pipeline Test

**Location:** `trapdata/tests/test_pipeline.py`

**Purpose:** Test complete ML pipeline from image import to tracking

**Flow:**
```python
def test_full_pipeline():
    """Test complete pipeline end-to-end"""

    # 1. Setup test database
    db = create_test_db()

    # 2. Import test images
    import_images(db, "trapdata/tests/images/denmark/")

    # 3. Verify images imported
    assert db.query(TrapImage).count() > 0

    # 4. Queue images
    queue_manager = QueueManager(db)
    queue_manager.add_images_to_queue()

    # 5. Run localization
    run_localization(db, settings)

    # 6. Verify detections created
    detections = db.query(DetectedObject).all()
    assert len(detections) > 0

    # 7. Run binary classification
    run_binary_classification(db, settings)

    # 8. Verify moths identified
    moths = db.query(DetectedObject).filter_by(
        binary_label="moth"
    ).all()
    assert len(moths) > 0

    # 9. Run species classification
    run_species_classification(db, settings)

    # 10. Verify species assigned
    classified = db.query(DetectedObject).filter(
        DetectedObject.specific_label.isnot(None)
    ).all()
    assert len(classified) > 0

    # 11. Run feature extraction
    run_feature_extraction(db, settings)

    # 12. Verify features extracted
    with_features = db.query(DetectedObject).filter(
        DetectedObject.cnn_features.isnot(None)
    ).all()
    assert len(with_features) > 0

    # 13. Run tracking
    run_tracking(db, settings)

    # 14. Verify tracks assigned
    tracked = db.query(DetectedObject).filter(
        DetectedObject.sequence_id.isnot(None)
    ).all()
    assert len(tracked) > 0

    # 15. Verify track continuity
    for track_id in get_unique_track_ids(db):
        track = get_track(db, track_id)
        verify_track_continuity(track)
```

**Running:**
```bash
# Via pytest
pytest trapdata/tests/test_pipeline.py -v

# Via CLI
ami test pipeline
```

## Integration Tests

### API Integration Tests

**Location:** `trapdata/api/tests/test_api.py`

**Test API endpoints with real models:**

```python
from fastapi.testclient import TestClient
from trapdata.api.api import app

client = TestClient(app)

def test_process_endpoint():
    """Test /process endpoint with real image"""

    # Load test image
    with open("trapdata/tests/images/panama/moth.jpg", "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    # Prepare request
    request_data = {
        "pipeline": "panama_moths_2023",
        "source_images": [
            {
                "id": "test_001",
                "image": f"data:image/jpeg;base64,{image_b64}"
            }
        ]
    }

    # Send request
    response = client.post("/process", json=request_data)

    # Verify response
    assert response.status_code == 200
    data = response.json()

    assert "source_images" in data
    assert len(data["source_images"]) == 1

    image_result = data["source_images"][0]
    assert "detections" in image_result
    assert len(image_result["detections"]) > 0

    # Check detection format
    detection = image_result["detections"][0]
    assert "bbox" in detection
    assert len(detection["bbox"]) == 4
    assert "classification" in detection
    assert "specific_label" in detection["classification"]
```

**Test error handling:**

```python
def test_invalid_pipeline():
    """Test error handling for invalid pipeline"""

    request_data = {
        "pipeline": "nonexistent_pipeline",
        "source_images": [
            {"id": "test", "image": "https://example.com/image.jpg"}
        ]
    }

    response = client.post("/process", json=request_data)
    assert response.status_code == 400
    assert "invalid pipeline" in response.json()["detail"].lower()

def test_missing_required_field():
    """Test validation error for missing field"""

    request_data = {
        "source_images": [{"id": "test", "image": "url"}]
        # Missing 'pipeline' field
    }

    response = client.post("/process", json=request_data)
    assert response.status_code == 422
```

### Database Integration Tests

**Location:** `trapdata/db/tests.py`

**Test database operations:**

```python
def test_monitoring_session_aggregates(db_session):
    """Test aggregate counts update correctly"""

    # Create session
    session = MonitoringSession(
        day=date(2024, 1, 1),
        base_directory="/test"
    )
    db_session.add(session)
    db_session.flush()

    # Add images
    for i in range(5):
        img = TrapImage(
            monitoring_session_id=session.id,
            path=f"image{i}.jpg"
        )
        db_session.add(img)

    db_session.commit()

    # Update aggregates
    session.update_aggregates()

    # Verify
    assert session.num_images == 5
    assert session.num_detected_objects == 0

def test_tracking_assignment(db_session):
    """Test tracking assignment works correctly"""

    # Create image
    img = TrapImage(monitoring_session_id=1, path="test.jpg")
    db_session.add(img)
    db_session.flush()

    # Create detections
    det1 = DetectedObject(
        image_id=img.id,
        bbox=[0, 0, 100, 100],
        sequence_id="track_1",
        sequence_frame=0
    )
    db_session.add(det1)
    db_session.flush()

    det2 = DetectedObject(
        image_id=img.id,
        bbox=[10, 10, 110, 110],
        sequence_id="track_1",
        sequence_frame=1,
        sequence_previous_id=det1.id
    )
    db_session.add(det2)
    db_session.commit()

    # Verify track
    track = db_session.query(DetectedObject).filter_by(
        sequence_id="track_1"
    ).order_by(DetectedObject.sequence_frame).all()

    assert len(track) == 2
    assert track[1].sequence_previous_id == track[0].id
```

## Unit Tests

### Model Registry Tests

**Location:** `trapdata/tests/test_registry.py`

```python
def test_model_registration():
    """Test models auto-register correctly"""

    from trapdata.ml.models.classification import SpeciesClassifier

    # Check registry populated
    assert len(SpeciesClassifier._registry) > 0

    # Check specific model registered
    assert "panama_moths_2023" in SpeciesClassifier._registry

def test_get_model():
    """Test model retrieval"""

    from trapdata.ml.models.classification import SpeciesClassifier

    # Get model class
    clf_class = SpeciesClassifier.get_model("panama_moths_2023")

    # Verify it's a class
    assert isinstance(clf_class, type)

    # Instantiate
    clf = clf_class()

    # Verify attributes
    assert clf.slug == "panama_moths_2023"
    assert hasattr(clf, 'category_map')
    assert len(clf.category_map) > 0
```

### EXIF Parsing Tests

**Location:** `trapdata/tests/test_exif.py`

```python
def test_exif_orientation():
    """Test EXIF orientation correction"""

    from trapdata.common.filemanagement import get_exif_orientation

    # Test image with orientation tag
    img_path = "trapdata/tests/images/rotated.jpg"
    orientation = get_exif_orientation(img_path)

    assert orientation in [1, 3, 6, 8]  # Valid orientation values

def test_exif_timestamp():
    """Test extracting timestamp from EXIF"""

    from trapdata.common.filemanagement import get_exif_timestamp

    img_path = "trapdata/tests/images/panama/moth.jpg"
    timestamp = get_exif_timestamp(img_path)

    assert timestamp is not None
    assert isinstance(timestamp, datetime)
```

### Utility Function Tests

```python
def test_bbox_area():
    """Test bounding box area calculation"""

    from trapdata.common.utils import calculate_bbox_area

    bbox = [10, 20, 50, 80]
    area = calculate_bbox_area(bbox)

    assert area == 40 * 60  # width * height
    assert area == 2400

def test_bbox_iou():
    """Test intersection over union"""

    from trapdata.common.utils import calculate_iou

    bbox1 = [0, 0, 100, 100]
    bbox2 = [50, 50, 150, 150]

    iou = calculate_iou(bbox1, bbox2)

    # Overlapping area: 50x50 = 2500
    # Union area: 10000 + 10000 - 2500 = 17500
    # IoU: 2500 / 17500 ≈ 0.143

    assert 0.14 <= iou <= 0.15
```

## Test Fixtures

### Database Fixture

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from trapdata.db.base import Base

@pytest.fixture
def db_session():
    """Create in-memory database for testing"""

    # Create engine
    engine = create_engine("sqlite:///:memory:")

    # Create tables
    Base.metadata.create_all(engine)

    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    # Cleanup
    session.close()
```

### Settings Fixture

```python
@pytest.fixture
def test_settings():
    """Create test settings"""

    from trapdata.settings import Settings

    settings = Settings(
        database_url="sqlite:///:memory:",
        image_base_path="trapdata/tests/images",
        localization_model="faster_rcnn_resnet50",
        species_classification_model="panama_moths_2023",
        classification_batch_size=8
    )

    return settings
```

### Test Images Fixture

```python
@pytest.fixture
def test_images():
    """Load test images"""

    from PIL import Image
    import glob

    images = []
    for path in glob.glob("trapdata/tests/images/panama/*.jpg"):
        img = Image.open(path)
        images.append(img)

    return images
```

## Mocking

### Mock Model Inference

```python
from unittest.mock import Mock, patch

def test_pipeline_with_mock_model():
    """Test pipeline with mocked model"""

    # Create mock model
    mock_model = Mock()
    mock_model.predict.return_value = [
        {
            'bbox': [0, 0, 100, 100],
            'confidence': 0.9,
            'label': 'moth'
        }
    ]

    # Patch model loading
    with patch('trapdata.ml.models.localization.ObjectDetector') as MockDetector:
        MockDetector.return_value = mock_model

        # Run pipeline
        run_pipeline(db, settings)

        # Verify mock was called
        assert mock_model.predict.called
```

### Mock File Operations

```python
def test_import_with_mock_filesystem():
    """Test import with mocked filesystem"""

    from unittest.mock import patch

    mock_images = [
        "/fake/path/image1.jpg",
        "/fake/path/image2.jpg"
    ]

    with patch('glob.glob', return_value=mock_images):
        with patch('os.path.exists', return_value=True):
            images = find_images("/fake/path")

    assert len(images) == 2
```

## CI/CD Testing

### GitHub Actions Workflow

**Location:** `.github/workflows/test.yml`

```yaml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'

    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -

    - name: Install dependencies
      run: |
        poetry install

    - name: Run tests
      run: |
        poetry run pytest --cov=trapdata --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

### ML-Specific Tests

**Location:** `.github/workflows/test-ml.yml`

```yaml
name: ML Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-ml:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        poetry install

    - name: Download test models
      run: |
        poetry run python scripts/download_test_models.py

    - name: Run ML tests
      run: |
        poetry run pytest trapdata/tests/test_pipeline.py -v
```

## Test Coverage

### Measure Coverage

```bash
# Run with coverage
pytest --cov=trapdata

# Generate HTML report
pytest --cov=trapdata --cov-report=html

# View report
open htmlcov/index.html
```

### Coverage Goals

- **Overall:** > 80%
- **Core modules:** > 90%
  - `trapdata.ml.pipeline`
  - `trapdata.db.models`
  - `trapdata.api.api`
- **Utilities:** > 95%
  - `trapdata.common.utils`

## Performance Testing

### Benchmark Pipeline

```python
import time

def test_pipeline_performance(db_session, test_settings):
    """Benchmark pipeline performance"""

    # Import images
    import_images(db_session, "trapdata/tests/images/denmark/")

    num_images = db_session.query(TrapImage).count()

    # Time pipeline
    start = time.time()
    run_pipeline(db_session, test_settings)
    end = time.time()

    elapsed = end - start
    images_per_second = num_images / elapsed

    print(f"Processed {num_images} images in {elapsed:.2f}s")
    print(f"Throughput: {images_per_second:.2f} images/second")

    # Assert reasonable performance
    assert images_per_second > 1.0  # At least 1 image/second
```

### Profile Tests

```bash
# Install pytest-profiling
pip install pytest-profiling

# Run with profiling
pytest --profile

# View profile
python -m pstats prof/combined.prof
```

## Best Practices

### 1. Isolate Tests

```python
# Bad: Tests depend on each other
def test_step1():
    global result
    result = process()

def test_step2():
    assert result.valid  # Depends on test_step1!

# Good: Each test is independent
def test_step1():
    result = process()
    assert result is not None

def test_step2():
    result = process()
    assert result.valid
```

### 2. Use Fixtures

```python
# Bad: Setup in every test
def test_detection():
    db = create_db()
    session = MonitoringSession(...)
    db.add(session)
    # ... test code

def test_classification():
    db = create_db()  # Duplicate setup
    session = MonitoringSession(...)
    db.add(session)
    # ... test code

# Good: Use fixture
@pytest.fixture
def session_with_data(db_session):
    session = MonitoringSession(...)
    db_session.add(session)
    return session

def test_detection(session_with_data):
    # Use session
    pass

def test_classification(session_with_data):
    # Use session
    pass
```

### 3. Test Edge Cases

```python
def test_classification_with_edge_cases():
    """Test classification handles edge cases"""

    # Empty image
    result = classify([])
    assert result == []

    # Single pixel image
    tiny_img = Image.new('RGB', (1, 1))
    result = classify([tiny_img])
    assert len(result) == 1

    # Very large image
    huge_img = Image.new('RGB', (10000, 10000))
    result = classify([huge_img])
    assert len(result) == 1

    # Invalid image data
    with pytest.raises(ValueError):
        classify([None])
```

### 4. Clear Test Names

```python
# Bad: Unclear what's being tested
def test_1():
    ...

def test_detection():
    ...

# Good: Descriptive names
def test_detection_creates_bounding_boxes():
    ...

def test_detection_filters_low_confidence():
    ...

def test_detection_handles_empty_image():
    ...
```

### 5. Arrange-Act-Assert

```python
def test_tracking_assigns_sequence_id():
    # Arrange
    det1 = DetectedObject(bbox=[0, 0, 100, 100])
    det2 = DetectedObject(bbox=[10, 10, 110, 110])

    # Act
    assign_tracking(det1, det2)

    # Assert
    assert det2.sequence_id == det1.sequence_id
    assert det2.sequence_previous_id == det1.id
```
