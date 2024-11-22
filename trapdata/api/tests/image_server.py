import http.server
import logging
import socketserver
import threading
import unittest
from functools import partial
from pathlib import Path
from typing import List

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestImageServer")


class TestImageServer:
    def __init__(self, test_images_dir: str | Path, port: int = 8000):
        self.test_images_dir = Path(test_images_dir)
        self.port = port
        self._server = None
        self._thread = None
        self.logger = logger.getChild("server")

        if not self.test_images_dir.exists():
            raise ValueError(
                f"Test images directory does not exist: {self.test_images_dir}"
            )

    def list_image_files(self) -> List[Path]:
        """Return a list of all image files in the test directory"""
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        return [
            f.relative_to(self.test_images_dir)
            for f in self.test_images_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

    def __enter__(self):
        self.logger.info(f"Starting test server for directory: {self.test_images_dir}")

        # Create custom handler that serves files from test directory
        handler = partial(
            http.server.SimpleHTTPRequestHandler,
            directory=str(self.test_images_dir),
            # log_function=lambda *args: None,  # Suppress default handler logging
        )

        # Find an available port if the specified one is in use
        while True:
            try:
                self._server = socketserver.TCPServer(("", self.port), handler)
                self.logger.info(f"Server started on port {self.port}")
                break
            except OSError:
                self.logger.debug(f"Port {self.port} in use, trying next port")
                self.port += 1

        # Start server in a separate thread
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True
        self._thread.start()
        self.logger.debug("Server thread started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Shutting down test server")
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join()
        self.logger.debug("Server shutdown complete")

    def get_url(self, path: str | Path) -> str:
        """Convert a local file path to its temporary server URL"""
        return f"http://localhost:{self.port}/{path}"


class TestImageProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Look for test_images relative to the test file location
        cls.test_file_dir = Path(__file__).parent.parent
        cls.test_images_dir = cls.test_file_dir / "tests/images/vermont"

        # Log test configuration
        cls.logger = logger.getChild("TestImageProcessing")
        cls.logger.info(f"Test images directory: {cls.test_images_dir}")

        if not cls.test_images_dir.exists():
            raise FileNotFoundError(
                f"Test images directory not found: {cls.test_images_dir}"
            )

    def setUp(self):
        self.server = TestImageServer(self.test_images_dir)
        self.logger = self.__class__.logger.getChild(self._testMethodName)

    def test_image_processing(self):
        with self.server:
            # Find available test images
            test_images = self.server.list_image_files()
            if not test_images:
                self.skipTest("No test images found in directory")

            # Test with first available image
            test_image = test_images[0]
            self.logger.info(f"Testing with image: {test_image}")

            image_url = self.server.get_url(test_image)
            self.logger.debug(f"Generated URL: {image_url}")

            # Your actual test code here
            response = requests.get(image_url)
            self.assertEqual(response.status_code, 200)
            self.logger.info(
                f"Successfully retrieved image: {len(response.content)} bytes"
            )

            # Example: Test your image processing function
            # processed_image = your_module.process_image(image_url)
            # self.assertIsNotNone(processed_image)

    def test_multiple_images(self):
        with self.server:
            # Find all test images
            test_images = self.server.list_image_files()
            if not test_images:
                self.skipTest("No test images found in directory")

            self.logger.info(f"Testing batch processing with {len(test_images)} images")
            image_urls = [self.server.get_url(img) for img in test_images]

            # Your batch processing test code here
            for url, image_path in zip(image_urls, test_images):
                self.logger.debug(f"Testing image: {image_path}")
                response = requests.get(url)
                self.assertEqual(response.status_code, 200)
                self.logger.debug(f"Successfully retrieved: {image_path}")


if __name__ == "__main__":
    unittest.main()
