"""CLI commands for Antenna worker.

This module provides thin CLI wrappers for the Antenna worker functionality.
All business logic has been moved to trapdata.antenna module.
"""

# Re-export functions for backwards compatibility with cli/base.py
from trapdata.antenna.registration import register_pipelines
from trapdata.antenna.worker import run_worker

__all__ = ["run_worker", "register_pipelines"]
