"""Unit tests for batched classification in worker.py.

This test validates that the worker correctly batches multiple detection crops
together for classification instead of processing them one at a time.
"""

import torch
import torchvision.transforms
from unittest.mock import MagicMock, patch


def test_batched_classification():
    """Test that worker batches all crops together for classification."""
    from trapdata.cli.worker import _process_job
    from trapdata.api.schemas import DetectionResponse, BBox
    
    # Mock the dataloader to return a batch with detections
    mock_detector_results = [
        DetectionResponse(
            source_image_id="img1",
            bbox=BBox(x1=10, y1=10, x2=50, y2=50),
            score=0.9,
            classifications=[],
        ),
        DetectionResponse(
            source_image_id="img1",
            bbox=BBox(x1=60, y1=60, x2=100, y2=100),
            score=0.85,
            classifications=[],
        ),
        DetectionResponse(
            source_image_id="img1",
            bbox=BBox(x1=110, y1=110, x2=150, y2=150),
            score=0.8,
            classifications=[],
        ),
    ]
    
    # Mock classifier
    mock_classifier = MagicMock()
    mock_classifier.get_transforms.return_value = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
    ])
    
    # Track how many times predict_batch is called and with what batch sizes
    predict_batch_calls = []
    
    def mock_predict_batch(batch):
        predict_batch_calls.append(batch.shape[0])  # Record batch size
        # Return dummy output for each item in batch
        return torch.rand(batch.shape[0], 10)  # 10 classes
    
    mock_classifier.predict_batch = mock_predict_batch
    mock_classifier.post_process_batch.return_value = [
        MagicMock(labels=["class1"] * 10, logit=[0.0] * 10, scores=[0.1] * 10)
        for _ in range(3)
    ]
    mock_classifier.update_detection_classification.return_value = mock_detector_results[0]
    mock_classifier.reset = MagicMock()
    
    # Mock detector
    mock_detector = MagicMock()
    mock_detector.results = mock_detector_results
    mock_detector.predict_batch.return_value = []
    mock_detector.post_process_batch.return_value = []
    mock_detector.save_results = MagicMock()
    mock_detector.reset = MagicMock()
    
    # Create a simple batch with one image
    image_tensor = torch.rand(3, 200, 200)
    mock_batch = {
        "images": [image_tensor],
        "image_ids": ["img1"],
        "reply_subjects": ["subj1"],
        "image_urls": ["http://example.com/img1.jpg"],
        "failed_items": [],
    }
    
    # Mock the dataloader to return our batch
    mock_loader = [mock_batch]
    
    # Mock settings
    mock_settings = MagicMock()
    mock_settings.antenna_api_base_url = "http://localhost:8000/api/v2"
    mock_settings.antenna_api_auth_token = "test_token"
    
    # Patch dependencies
    with patch("trapdata.cli.worker.get_rest_dataloader", return_value=mock_loader), \
         patch("trapdata.cli.worker.CLASSIFIER_CHOICES", {"test_pipeline": MagicMock(return_value=mock_classifier)}), \
         patch("trapdata.cli.worker.APIMothDetector", return_value=mock_detector), \
         patch("trapdata.cli.worker.post_batch_results", return_value=True):
        
        # Run the worker
        _process_job(pipeline="test_pipeline", job_id=1, settings=mock_settings)
    
    # Verify that predict_batch was called exactly once (batched)
    assert len(predict_batch_calls) == 1, (
        f"Expected predict_batch to be called once (batched), "
        f"but it was called {len(predict_batch_calls)} times"
    )
    
    # Verify the batch size was 3 (all crops together)
    assert predict_batch_calls[0] == 3, (
        f"Expected batch size of 3, but got {predict_batch_calls[0]}"
    )
    
    print("✓ Batched classification test passed!")
    print(f"  - predict_batch called {len(predict_batch_calls)} time(s)")
    print(f"  - Batch size: {predict_batch_calls[0]} crops")


if __name__ == "__main__":
    test_batched_classification()
