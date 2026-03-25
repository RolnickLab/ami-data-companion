# ML Pipeline Documentation

## Pipeline Overview

The ML pipeline processes camera trap images through five sequential stages to detect, classify, and track insects.

### Pipeline Stages

```
┌──────────────────────────────────────────────────────────────┐
│  1. LOCALIZATION (Object Detection)                          │
│     Input: Raw images                                        │
│     Model: FasterRCNN (ResNet50 backbone)                    │
│     Output: Bounding boxes + confidence scores               │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  2. BINARY CLASSIFICATION (moth/non-moth filter)             │
│     Input: Cropped detections                                │
│     Model: Binary CNN classifier                             │
│     Output: "moth" or "nonmoth" + confidence                 │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  3. SPECIES CLASSIFICATION                                   │
│     Input: Moth detections only                              │
│     Model: Multi-class CNN (EfficientNet/ResNet)             │
│     Output: Species name + confidence scores                 │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  4. FEATURE EXTRACTION                                       │
│     Input: Classified detections                             │
│     Model: CNN feature extractor                             │
│     Output: 512-dim feature vector                           │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  5. TRACKING (Cross-frame matching)                          │
│     Input: Features from sequential frames                   │
│     Algorithm: Cosine similarity + Hungarian matching        │
│     Output: sequence_id linking detections                   │
└──────────────────────────────────────────────────────────────┘
```

## Stage 1: Localization (Object Detection)

### Purpose
Detect all potential insects in an image and generate bounding boxes.

### Model Architecture
- **Base Model:** FasterRCNN with ResNet50 backbone
- **Pre-training:** COCO dataset
- **Fine-tuning:** AMI moth dataset
- **Output:** Bounding boxes `[x1, y1, x2, y2]` + confidence scores

### Implementation
```python
# trapdata/ml/models/localization.py
class ObjectDetector(InferenceBaseClass):
    def load_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False,
            num_classes=2  # background + moth
        )
        model.load_state_dict(torch.load(self.model_path))
        return model

    def predict(self, images):
        """
        Args:
            images: List of PIL Images or torch tensors

        Returns:
            List of dicts with 'boxes', 'scores', 'labels'
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        return predictions
```

### Key Parameters
- **Confidence Threshold:** 0.5 (default) - Minimum score to keep detection
- **NMS Threshold:** 0.3 - Non-maximum suppression threshold
- **Max Detections:** 100 - Maximum detections per image

### Performance
- **Input Size:** Variable (resized to max 1333px)
- **Batch Size:** 8-16 images (GPU dependent)
- **Speed:** ~0.5-1.0 seconds per image on GPU

### Gotchas
- **Image orientation:** EXIF orientation must be corrected before inference
- **Memory:** Large images can cause OOM - resize if needed
- **Batch processing:** Images in batch should have similar sizes

## Stage 2: Binary Classification

### Purpose
Filter out non-moth detections (e.g., spider webs, debris, other insects).

### Model Architecture
- **Architecture:** ResNet18 or EfficientNet-B0
- **Input Size:** 224x224 pixels
- **Output:** 2 classes ("moth", "nonmoth")

### Implementation
```python
# trapdata/ml/models/classification.py
class BinaryClassifier(InferenceBaseClass):
    category_map = {
        0: "nonmoth",
        1: "moth"
    }

    def predict(self, crops):
        """
        Args:
            crops: List of PIL Images (cropped detections)

        Returns:
            List of dicts with 'label', 'score', 'logits'
        """
        # Preprocess
        inputs = torch.stack([self.transform(crop) for crop in crops])

        # Inference
        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)

        # Format results
        results = []
        for prob, logit in zip(probs, logits):
            class_idx = prob.argmax().item()
            results.append({
                'label': self.category_map[class_idx],
                'score': prob[class_idx].item(),
                'logits': logit.tolist()
            })
        return results
```

### Key Parameters
- **Threshold:** 0.5 - Minimum confidence to classify as "moth"
- **Batch Size:** 64-128 crops

### Performance
- **Speed:** ~100-200 crops/second on GPU
- **Accuracy:** ~95% on validation set

### Gotchas
- **Class imbalance:** More moths than non-moths in training data
- **Edge cases:** Very small or very large detections may misclassify
- **Lighting:** Poor lighting affects performance

## Stage 3: Species Classification

### Purpose
Identify the species of detected moths.

### Available Classifiers

**Regional Models:**
- `panama_moths_2023` - Panama dataset (87 species)
- `panama_moths_2024` - Updated Panama model (120 species)
- `quebec_vermont_moths_2023` - Quebec/Vermont (200+ species)
- `uk_denmark_moths_2023` - UK/Denmark (150+ species)
- `costa_rica_moths_turing_2024` - Costa Rica (Turing dataset)

**Global Models:**
- `global_moths_2024` - Multi-region model (500+ species)
- `insect_orders_2025` - Order-level classification (Lepidoptera, Coleoptera, etc.)

### Model Architecture
- **Architecture:** EfficientNet-B3 or ResNet50
- **Input Size:** 224x224 or 384x384 pixels
- **Output:** N-class softmax (N = number of species)

### Implementation
```python
class SpeciesClassifier(InferenceBaseClass):
    slug = "uk_denmark_moths_2023"
    name = "UK/Denmark Moth Classifier"
    version = "1.0"

    category_map = {
        0: "Actias luna",
        1: "Autographa gamma",
        # ... more species
    }

    def predict(self, crops):
        # Similar to binary classifier
        # Returns species name + confidence + logits
        pass
```

### Key Parameters
- **Threshold:** 0.3-0.5 - Minimum confidence to accept classification
- **Top-K:** 5 - Return top 5 predictions
- **Batch Size:** 32-64 crops

### Performance
- **Speed:** ~50-100 crops/second on GPU (depends on model size)
- **Accuracy:** Varies by model (60-90% top-1, 80-95% top-5)

### Gotchas
- **Taxonomic accuracy:** Species names must match taxonomy database
- **Unknown species:** Models can't classify species not in training set
- **Similar species:** Closely related species hard to distinguish
- **Image quality:** Blurry or partial moths harder to classify

## Stage 4: Feature Extraction

### Purpose
Extract CNN features for tracking across frames.

### Model Architecture
- **Architecture:** ResNet50 (pretrained on ImageNet)
- **Layer:** Last conv layer before final FC (2048-dim)
- **Output:** 512-dim feature vector (after PCA/projection)

### Implementation
```python
# trapdata/ml/models/tracking.py
class FeatureExtractor(InferenceBaseClass):
    def load_model(self):
        # Load ResNet50
        model = torchvision.models.resnet50(pretrained=True)

        # Remove final FC layer
        self.model = nn.Sequential(*list(model.children())[:-1])

        # Add projection to 512-dim
        self.projection = nn.Linear(2048, 512)

    def extract_features(self, crops):
        """
        Args:
            crops: List of PIL Images

        Returns:
            np.ndarray of shape (N, 512) - Feature vectors
        """
        inputs = torch.stack([self.transform(crop) for crop in crops])

        with torch.no_grad():
            features = self.model(inputs)
            features = features.squeeze()
            features = self.projection(features)

        return features.cpu().numpy()
```

### Key Parameters
- **Feature Dimension:** 512 (good balance of size and discriminability)
- **Normalization:** L2 normalization for cosine similarity
- **Batch Size:** 64-128 crops

### Performance
- **Speed:** ~100-200 crops/second on GPU
- **Memory:** ~2KB per detection (512 floats)

## Stage 5: Tracking

### Purpose
Link detections across frames to identify individual insects.

### Algorithm
1. **Extract features** from current and previous frame detections
2. **Compute similarity matrix** using cosine similarity
3. **Hungarian algorithm** for optimal assignment
4. **Threshold filtering** - Only accept matches with similarity > threshold
5. **Assign sequence IDs** to tracked individuals

### Implementation
```python
# trapdata/ml/models/tracking.py
def track_detections(current_detections, previous_detections):
    """
    Track detections across frames using feature similarity.

    Args:
        current_detections: Detections with CNN features
        previous_detections: Detections from previous frame

    Returns:
        Assignments: Dict mapping current_id -> previous_id
    """
    # Extract feature vectors
    current_features = np.array([d.cnn_features for d in current_detections])
    previous_features = np.array([d.cnn_features for d in previous_detections])

    # Compute cosine similarity matrix
    similarity = cosine_similarity(current_features, previous_features)

    # Convert to cost (lower is better)
    cost_matrix = 1 - similarity

    # Hungarian algorithm for optimal assignment
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter by threshold
    assignments = {}
    for i, j in zip(row_ind, col_ind):
        cost = cost_matrix[i, j]
        if cost < TRACKING_COST_THRESHOLD:  # Default: 1.0
            assignments[current_detections[i].id] = previous_detections[j].id

    return assignments
```

### Key Parameters
- **Cost Threshold:** 1.0 - Maximum cost to accept match (lower = more strict)
- **Temporal Gap:** 1 frame - Only match to immediately previous frame
- **Feature Similarity:** Cosine similarity (range: -1 to 1)

### Performance
- **Speed:** ~0.01 seconds for 100 detections
- **Memory:** Minimal (similarity matrix is small)

### Tracking Logic
```python
# Assign sequence IDs
for current_det in current_detections:
    if current_det.id in assignments:
        # Matched to previous detection
        prev_det_id = assignments[current_det.id]
        prev_det = get_detection(prev_det_id)

        current_det.sequence_id = prev_det.sequence_id
        current_det.sequence_frame = prev_det.sequence_frame + 1
        current_det.sequence_previous_id = prev_det.id
        current_det.sequence_previous_cost = cost_matrix[i, j]
    else:
        # New track
        current_det.sequence_id = f"track_{uuid.uuid4()}"
        current_det.sequence_frame = 0
        current_det.sequence_previous_id = None
```

### Gotchas
- **Frame gaps:** If frames are missing, tracking breaks
- **Multiple individuals:** Same species in frame can swap tracks
- **Occlusion:** Temporarily hidden insects restart as new tracks
- **Similar appearances:** Visually similar moths may link incorrectly

## Pipeline Orchestration

### Full Pipeline Execution
```python
# trapdata/ml/pipeline.py
def run_pipeline(db_session, settings):
    """Run full pipeline on queued images"""

    # Stage 1: Localization
    images = get_queued_images(db_session, batch_size=32)
    detector = ObjectDetector(settings.localization_model)
    for batch in images:
        results = detector.predict(batch)
        save_detections(db_session, results)

    # Stage 2: Binary Classification
    detections = get_unclassified_detections(db_session, batch_size=64)
    binary_clf = BinaryClassifier(settings.binary_classification_model)
    for batch in detections:
        crops = load_crops(batch)
        results = binary_clf.predict(crops)
        save_classifications(db_session, batch, results)

    # Stage 3: Species Classification (moths only)
    moths = get_moths_without_species(db_session, batch_size=64)
    species_clf = SpeciesClassifier(settings.species_classification_model)
    for batch in moths:
        crops = load_crops(batch)
        results = species_clf.predict(crops)
        save_species(db_session, batch, results)

    # Stage 4: Feature Extraction
    detections = get_detections_without_features(db_session, batch_size=64)
    feature_extractor = FeatureExtractor(settings.feature_extractor)
    for batch in detections:
        crops = load_crops(batch)
        features = feature_extractor.extract_features(crops)
        save_features(db_session, batch, features)

    # Stage 5: Tracking
    for session in get_monitoring_sessions(db_session):
        track_session(db_session, session)
```

### Partial Pipeline Execution
```python
# Run only localization
run_localization_only(db_session)

# Run classification on existing detections
run_classification_only(db_session)

# Retrack with new parameters
retrack_all(db_session, cost_threshold=0.8)
```

## Model Registry

### Auto-Registration Pattern
```python
# Models auto-register on import
from trapdata.ml.models.classification import MothClassifierPanama

# Access via registry
classifier_class = SpeciesClassifier.get_model("panama_moths_2023")
classifier = classifier_class()
```

### Adding New Models
```python
class MyNewClassifier(SpeciesClassifier):
    slug = "my_region_2025"  # Unique identifier
    name = "My Region Moth Classifier"
    version = "1.0"
    model_url = "https://example.com/model.pth"

    category_map = {
        0: "Species A",
        1: "Species B",
        # ...
    }

    def __init__(self):
        super().__init__()
        # Custom initialization
```

## Model Management

### Model Download
```python
# Models downloaded on first use
classifier = SpeciesClassifier("panama_moths_2023")
# Downloads model to: ~/.config/trapdata/models/panama_moths_2023.pth

# Manual download
from trapdata.ml.utils import download_model
download_model(url, destination)
```

### Model Caching
- Models loaded once and kept in memory
- Reused across batches
- Cleared when pipeline finishes

### User Data Directory
**Default locations:**
- macOS: `/Library/Application Support/trapdata/models/`
- Linux: `~/.config/trapdata/models/`
- Windows: `%AppData%/trapdata/models/`

## Performance Optimization

### Batch Processing
```python
# Bad: Process one at a time
for image in images:
    result = model.predict([image])

# Good: Process in batches
batch_size = 32
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    results = model.predict(batch)
```

### GPU Utilization
```python
# Auto-detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Monitor GPU usage
nvidia-smi

# Adjust batch size based on memory
# If OOM: reduce batch size
# If GPU underutilized: increase batch size
```

### DataLoader Optimization
```python
# Use PyTorch DataLoader for efficient loading
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel image loading
    pin_memory=True  # Faster GPU transfer
)

for batch in loader:
    results = model(batch)
```

## Testing ML Pipeline

### Unit Tests
```python
# Test model loading
def test_model_loads():
    clf = SpeciesClassifier("panama_moths_2023")
    assert clf.model is not None

# Test prediction format
def test_prediction_format():
    clf = SpeciesClassifier("panama_moths_2023")
    crops = load_test_images()
    results = clf.predict(crops)

    assert len(results) == len(crops)
    assert all('label' in r for r in results)
    assert all('score' in r for r in results)
```

### Integration Tests
```python
# trapdata/tests/test_pipeline.py
def test_full_pipeline():
    """Test complete pipeline end-to-end"""
    # Setup test database
    db = create_test_db()

    # Import test images
    import_images(db, "trapdata/tests/images/denmark/")

    # Run pipeline
    run_pipeline(db, test_settings)

    # Verify results
    detections = db.query(DetectedObject).all()
    assert len(detections) > 0
    assert all(d.specific_label is not None for d in detections)

    # Verify tracking
    tracks = db.query(DetectedObject.sequence_id).distinct().all()
    assert len(tracks) > 0
```

## Troubleshooting

### Common Issues

**1. OOM (Out of Memory):**
- Reduce batch size
- Resize large images
- Clear GPU cache: `torch.cuda.empty_cache()`

**2. Slow inference:**
- Check GPU is being used: `torch.cuda.is_available()`
- Increase batch size (if memory allows)
- Use DataLoader with num_workers > 0

**3. Poor detection accuracy:**
- Check image quality
- Verify EXIF orientation correction
- Try different confidence threshold

**4. Tracking breaks:**
- Check for frame gaps (missing images)
- Adjust tracking cost threshold
- Verify features are being extracted

**5. Model download fails:**
- Check internet connection
- Verify URL is correct
- Check user data directory permissions
