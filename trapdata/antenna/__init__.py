"""Antenna platform integration module.

This module provides integration with the Antenna platform for remote image processing.
It includes:
- API client for fetching jobs and posting results
- Worker loop for continuous job processing
- Pipeline registration with Antenna projects
- Schemas for Antenna API requests/responses
- Dataset classes for streaming tasks from the API
"""

from trapdata.antenna import client, datasets, registration, schemas, worker

__all__ = [
    "client",
    "datasets",
    "registration",
    "schemas",
    "worker",
]
