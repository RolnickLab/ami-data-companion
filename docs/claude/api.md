# API Documentation

## FastAPI REST API

The AMI Data Companion provides a REST API for remote image processing.

### Starting the API Server

```bash
# Development
ami api

# Production with uvicorn
uvicorn trapdata.api.api:app --host 0.0.0.0 --port 2000 --workers 4

# With custom settings
AMI_SPECIES_CLASSIFICATION_MODEL=global_moths_2024 ami api
```

**Default URL:** `http://localhost:2000`

## Endpoints

### POST /process

Process images through the ML pipeline.

**Request:**
```json
{
  "pipeline": "uk_denmark_moths_2023",
  "source_images": [
    {
      "id": "image_001",
      "image": "https://example.com/moth.jpg",
      "timestamp": "2024-01-15T22:30:00Z"
    }
  ],
  "config": {
    "localization_threshold": 0.5,
    "classification_threshold": 0.3
  }
}
```

**Fields:**
- `pipeline` (string, required): Pipeline slug (e.g., "uk_denmark_moths_2023")
- `source_images` (array, required): List of images to process
  - `id` (string, required): User-provided image identifier
  - `image` (string, required): Image URL or base64-encoded data
  - `timestamp` (string, optional): ISO 8601 timestamp
- `config` (object, optional): Pipeline configuration overrides

**Response:**
```json
{
  "source_images": [
    {
      "id": "image_001",
      "detections": [
        {
          "bbox": [100, 150, 300, 400],
          "confidence": 0.95,
          "classification": {
            "binary_label": "moth",
            "binary_score": 0.98,
            "specific_label": "Actias luna",
            "specific_score": 0.87,
            "top_5": [
              {"label": "Actias luna", "score": 0.87},
              {"label": "Actias selene", "score": 0.09}
            ]
          },
          "algorithm": "FasterRCNN-ResNet50",
          "timestamp": "2024-01-15T22:35:12Z"
        }
      ]
    }
  ],
  "pipeline": {
    "name": "UK/Denmark Moths 2023",
    "version": "1.0",
    "models": {
      "localization": "FasterRCNN-ResNet50",
      "binary_classification": "ResNet18",
      "species_classification": "EfficientNet-B3"
    }
  },
  "processing_time": 2.34,
  "total_detections": 12
}
```

**Status Codes:**
- `200 OK`: Successful processing
- `400 Bad Request`: Invalid request format
- `404 Not Found`: Pipeline not found
- `500 Internal Server Error`: Processing failed

### GET /info

Get service information and available pipelines.

**Request:**
```bash
curl http://localhost:2000/info
```

**Response:**
```json
{
  "service": "AMI Data Companion API",
  "version": "0.6.0",
  "available_pipelines": [
    {
      "slug": "panama_moths_2023",
      "name": "Panama Moth Classifier",
      "version": "1.0",
      "num_species": 87,
      "algorithms": {
        "localization": "FasterRCNN-ResNet50",
        "binary_classification": "ResNet18",
        "species_classification": "EfficientNet-B3"
      },
      "category_map": {
        "0": "Acharia stimulea",
        "1": "Actias luna",
        ...
      }
    },
    ...
  ]
}
```

### GET /readyz

Readiness check - are models loaded and ready?

**Request:**
```bash
curl http://localhost:2000/readyz
```

**Response:**
```json
{
  "ready": true,
  "pipelines": [
    "panama_moths_2023",
    "uk_denmark_moths_2023",
    "global_moths_2024"
  ]
}
```

### GET /livez

Liveness check - is service running?

**Request:**
```bash
curl http://localhost:2000/livez
```

**Response:**
```json
{
  "status": true
}
```

## Pydantic Schemas

### Request Schemas

**SourceImageRequest:**
```python
class SourceImageRequest(BaseModel):
    id: str
    image: str  # URL or base64
    timestamp: Optional[datetime] = None
```

**PipelineConfigRequest:**
```python
class PipelineConfigRequest(BaseModel):
    localization_threshold: float = 0.5
    classification_threshold: float = 0.3
    binary_threshold: float = 0.5
    max_detections: int = 100
```

**PipelineRequest:**
```python
class PipelineRequest(BaseModel):
    pipeline: str
    source_images: List[SourceImageRequest]
    config: Optional[PipelineConfigRequest] = None
```

### Response Schemas

**ClassificationResponse:**
```python
class ClassificationResponse(BaseModel):
    binary_label: str  # "moth" or "nonmoth"
    binary_score: float
    specific_label: Optional[str]
    specific_score: Optional[float]
    top_5: List[Dict[str, float]]
    logits: Optional[List[float]]
    timestamp: datetime
```

**DetectionResponse:**
```python
class DetectionResponse(BaseModel):
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    classification: ClassificationResponse
    algorithm: str
    timestamp: datetime
```

**SourceImageResponse:**
```python
class SourceImageResponse(BaseModel):
    id: str
    detections: List[DetectionResponse]
    width: int
    height: int
    processing_time: float
```

**PipelineResponse:**
```python
class PipelineResponse(BaseModel):
    source_images: List[SourceImageResponse]
    pipeline: Dict[str, Any]
    processing_time: float
    total_detections: int
```

## Usage Examples

### Python Client

```python
import requests
import base64

# Load image as base64
with open("moth.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Prepare request
request_data = {
    "pipeline": "uk_denmark_moths_2023",
    "source_images": [
        {
            "id": "moth_001",
            "image": f"data:image/jpeg;base64,{image_b64}",
            "timestamp": "2024-01-15T22:30:00Z"
        }
    ],
    "config": {
        "classification_threshold": 0.3
    }
}

# Send request
response = requests.post(
    "http://localhost:2000/process",
    json=request_data
)

# Parse response
result = response.json()
for image in result["source_images"]:
    print(f"Image: {image['id']}")
    for detection in image["detections"]:
        species = detection["classification"]["specific_label"]
        score = detection["classification"]["specific_score"]
        print(f"  - {species} ({score:.2%})")
```

### JavaScript/TypeScript Client

```typescript
async function processImage(imageUrl: string, pipeline: string) {
  const response = await fetch('http://localhost:2000/process', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      pipeline: pipeline,
      source_images: [
        {
          id: 'image_001',
          image: imageUrl,
          timestamp: new Date().toISOString(),
        },
      ],
    }),
  });

  const result = await response.json();
  return result;
}

// Usage
processImage('https://example.com/moth.jpg', 'uk_denmark_moths_2023')
  .then(result => {
    console.log(`Found ${result.total_detections} detections`);
    result.source_images.forEach(img => {
      img.detections.forEach(det => {
        console.log(`${det.classification.specific_label}: ${det.classification.specific_score}`);
      });
    });
  });
```

### cURL Examples

```bash
# Process image from URL
curl -X POST http://localhost:2000/process \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline": "uk_denmark_moths_2023",
    "source_images": [
      {
        "id": "test_001",
        "image": "https://example.com/moth.jpg"
      }
    ]
  }'

# Get available pipelines
curl http://localhost:2000/info | jq '.available_pipelines[].slug'

# Check if service is ready
curl http://localhost:2000/readyz
```

## Available Pipelines

### Regional Models

**panama_moths_2023:**
- Species: 87
- Region: Panama
- Use case: Tropical moths in Central America

**quebec_vermont_moths_2023:**
- Species: 200+
- Region: Quebec, Vermont
- Use case: Temperate moths in Northeast North America

**uk_denmark_moths_2023:**
- Species: 150+
- Region: UK, Denmark
- Use case: European moths

**costa_rica_moths_turing_2024:**
- Species: Varies
- Region: Costa Rica
- Use case: Turing dataset moths

### Global Models

**global_moths_2024:**
- Species: 500+
- Region: Multi-region
- Use case: General-purpose moth classification

**insect_orders_2025:**
- Classes: Insect orders (Lepidoptera, Coleoptera, etc.)
- Region: Global
- Use case: High-level insect classification

### Binary Model

**moth_binary:**
- Classes: 2 (moth, non-moth)
- Use case: Detection filtering without species ID

## Configuration

### Environment Variables

```bash
# API Configuration
AMI_API_HOST=0.0.0.0
AMI_API_PORT=2000
AMI_API_WORKERS=4

# Model Selection
AMI_SPECIES_CLASSIFICATION_MODEL=uk_denmark_moths_2023
AMI_BINARY_CLASSIFICATION_MODEL=moth_binary_2023
AMI_LOCALIZATION_MODEL=faster_rcnn_resnet50

# Performance
AMI_LOCALIZATION_BATCH_SIZE=16
AMI_CLASSIFICATION_BATCH_SIZE=64
AMI_NUM_WORKERS=4

# Thresholds
AMI_CLASSIFICATION_THRESHOLD=0.3
AMI_LOCALIZATION_THRESHOLD=0.5
```

### Runtime Configuration

Override settings per request:

```json
{
  "pipeline": "uk_denmark_moths_2023",
  "config": {
    "localization_threshold": 0.6,
    "classification_threshold": 0.4,
    "max_detections": 50
  }
}
```

## Performance Considerations

### Batch Processing

Process multiple images in one request for better throughput:

```json
{
  "pipeline": "uk_denmark_moths_2023",
  "source_images": [
    {"id": "img_001", "image": "url1"},
    {"id": "img_002", "image": "url2"},
    {"id": "img_003", "image": "url3"}
  ]
}
```

### Image Size

- **Recommended:** 1000-2000px on longest side
- **Maximum:** 4000px (larger images auto-resized)
- **Minimum:** 300px for good accuracy

### Concurrency

- API supports concurrent requests
- Model inference is thread-safe
- GPU shared across requests (batching helps)

### Caching

- Models loaded once on startup
- Kept in memory for fast inference
- No response caching (images vary)

## Error Handling

### Common Errors

**400 Bad Request:**
```json
{
  "detail": "Invalid pipeline: invalid_pipeline_name"
}
```

**422 Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "pipeline"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Model inference failed: CUDA out of memory"
}
```

### Error Recovery

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure retries
session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)

# Robust request
try:
    response = session.post(url, json=data, timeout=30)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

## Security

### Authentication

Currently no authentication (designed for local/trusted use).

For production deployment, add:
- API keys
- OAuth2
- IP whitelisting
- Rate limiting

### Input Validation

- Image URLs validated (no file:// or dangerous protocols)
- Base64 images size-limited
- Pipeline names validated against whitelist
- Config values validated by Pydantic

### CORS

Enable CORS for web clients:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Monitoring

### Logging

API logs requests and errors:

```python
import structlog
logger = structlog.get_logger()

logger.info("Processing request",
    pipeline=pipeline,
    num_images=len(source_images),
    request_id=request_id)
```

### Metrics

Track key metrics:
- Requests per second
- Processing time per image
- Detection counts
- Error rates

### Health Checks

Use `/livez` and `/readyz` for health monitoring:

```bash
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /livez
    port: 2000

# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /readyz
    port: 2000
```

## Testing

### Unit Tests

```python
# trapdata/api/tests/test_api.py
from fastapi.testclient import TestClient
from trapdata.api.api import app

client = TestClient(app)

def test_info_endpoint():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "available_pipelines" in data

def test_process_endpoint():
    request_data = {
        "pipeline": "panama_moths_2023",
        "source_images": [
            {
                "id": "test",
                "image": "https://example.com/moth.jpg"
            }
        ]
    }
    response = client.post("/process", json=request_data)
    assert response.status_code == 200
```

### Integration Tests

```bash
# Start test server
ami api &
API_PID=$!

# Run tests
pytest trapdata/api/tests/

# Cleanup
kill $API_PID
```

## Deployment

### Docker

```dockerfile
FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install poetry
RUN poetry install --no-dev

EXPOSE 2000
CMD ["poetry", "run", "uvicorn", "trapdata.api.api:app", "--host", "0.0.0.0", "--port", "2000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "2000:2000"
    environment:
      - AMI_DATABASE_URL=postgresql://user:pass@db/ami
      - AMI_SPECIES_CLASSIFICATION_MODEL=global_moths_2024
    volumes:
      - model_cache:/root/.config/trapdata
    depends_on:
      - db

  db:
    image: postgres:14
    environment:
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  model_cache:
  postgres_data:
```

### Production Checklist

- [ ] Configure appropriate workers (CPU cores * 2)
- [ ] Set up HTTPS/TLS
- [ ] Add authentication
- [ ] Configure logging
- [ ] Set up monitoring
- [ ] Configure rate limiting
- [ ] Set resource limits (memory, CPU)
- [ ] Test error handling
- [ ] Document API for users
