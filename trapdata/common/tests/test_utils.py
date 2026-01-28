"""Tests for trapdata.common.utils module."""

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from trapdata.common.utils import get_http_session


class TestGetHttpSession:
    """Test the get_http_session() utility function."""

    def test_creates_session(self):
        """Test that get_http_session creates a requests.Session."""
        session = get_http_session()
        assert isinstance(session, requests.Session)

    def test_mounts_http_adapter(self):
        """Test that HTTPAdapter is mounted for both http and https."""
        session = get_http_session()

        # Check that adapters are mounted
        assert "http://" in session.adapters
        assert "https://" in session.adapters

        # Check that they are HTTPAdapter instances
        assert isinstance(session.adapters["http://"], HTTPAdapter)
        assert isinstance(session.adapters["https://"], HTTPAdapter)

    def test_retry_configuration(self):
        """Test that retry configuration is correctly set."""
        max_retries = 5
        backoff_factor = 1.0
        status_forcelist = (500, 502, 503, 504)

        session = get_http_session(
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )

        # Get the retry object from the adapter
        adapter = session.adapters["https://"]
        retry = adapter.max_retries

        assert isinstance(retry, Retry)
        assert retry.total == max_retries
        assert retry.backoff_factor == backoff_factor
        assert retry.status_forcelist == status_forcelist

    def test_default_values(self):
        """Test that default values are sensible."""
        session = get_http_session()

        adapter = session.adapters["https://"]
        retry = adapter.max_retries

        # Check defaults
        assert retry.total == 3
        assert retry.backoff_factor == 0.5
        assert retry.status_forcelist == (500, 502, 503, 504)

    def test_allowed_methods(self):
        """Test that allowed methods include GET and POST."""
        session = get_http_session()

        adapter = session.adapters["https://"]
        retry = adapter.max_retries

        # Check that allowed methods include GET and POST
        assert "GET" in retry.allowed_methods
        assert "POST" in retry.allowed_methods

    def test_raise_on_status_false(self):
        """Test that raise_on_status is False (let caller handle status codes)."""
        session = get_http_session()

        adapter = session.adapters["https://"]
        retry = adapter.max_retries

        # raise_on_status should be False to let the caller handle status codes
        assert retry.raise_on_status is False
