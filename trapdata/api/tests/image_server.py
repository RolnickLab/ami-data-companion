import logging
import socket
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class StaticFileTestServer:
    def __init__(self, test_images_dir: Union[str, Path]):
        self.test_images_dir = Path(test_images_dir)
        self.logger = logger.getChild("server")

        if not self.test_images_dir.exists():
            raise ValueError(
                f"Test images directory does not exist: {self.test_images_dir}"
            )

        # Create handler class with custom directory
        handler = partial(SimpleHTTPRequestHandler, directory=str(self.test_images_dir))

        # Find an available port
        self.port = self._get_free_port()
        self.server = HTTPServer(("localhost", self.port), handler)
        self.server_thread = None

    @staticmethod
    def _get_free_port():
        """Find a free port to use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def __enter__(self):
        self.logger.info(f"Starting test server for directory: {self.test_images_dir}")
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True  # Don't hang on exit
        self.server_thread.start()
        self.logger.info(f"Server started at http://localhost:{self.port}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Shutting down test server")
        self.server.shutdown()
        self.server.server_close()
        if self.server_thread:
            self.server_thread.join()

    def get_url(self, path: Union[str, Path]) -> str:
        """Convert a local file path to its temporary server URL"""
        return f"http://localhost:{self.port}/{path}"