"""Regression tests for the DataLoader settings used by the Antenna worker.

``get_rest_dataloader()`` must never use persistent workers (they leak pipes,
see #145), must apply the ``timeout`` setting (a dead worker subprocess would
otherwise hang the job forever, see #140), and must apply the configured start
method. The pipe count itself is measured in ``test_dataloader_pipe_leak.py``;
this file checks the configuration, so it also runs on machines without CUDA.
"""

from types import SimpleNamespace
from unittest import TestCase

import pytest

from trapdata.antenna.datasets import get_rest_dataloader


def _make_settings(**overrides) -> SimpleNamespace:
    """Build a minimal duck-typed Settings object.

    Using SimpleNamespace (not MagicMock) so that attribute access on a
    *missing* field raises AttributeError — that lets the test catch a
    typo'd setting name on the production side rather than swallowing it.
    """
    defaults = {
        "antenna_api_base_url": "http://testserver/api/v2",
        "antenna_api_auth_token": "test-token",
        "antenna_api_batch_size": 2,
        "num_workers": 0,
        "antenna_api_dataloader_mp_context": "",
        "antenna_api_dataloader_timeout_s": 300,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestDataLoaderHygieneDefaults(TestCase):
    """The default config must apply both new knobs."""

    def test_num_workers_zero_does_not_set_mp_context(self):
        """num_workers=0 = no subprocesses, so mp_context must stay None.

        Setting multiprocessing_context on a num_workers=0 DataLoader is a
        no-op at best and a TypeError in some torch versions.
        """
        loader = get_rest_dataloader(job_id=1, settings=_make_settings(num_workers=0))
        assert loader.multiprocessing_context is None
        assert loader.timeout == 0  # 0 = no timeout, same as PyTorch default

    def test_num_workers_positive_uses_pytorch_default_context_by_default(self):
        """The default start method is PyTorch's own, so no context is passed."""
        loader = get_rest_dataloader(job_id=1, settings=_make_settings(num_workers=1))
        assert loader.multiprocessing_context is None

    def test_workers_are_not_persistent(self):
        """Persistent workers leak two pipes per worker per job when pin_memory is
        on with CUDA, because PyTorch keeps them in an atexit hook (#145)."""
        loader = get_rest_dataloader(job_id=1, settings=_make_settings(num_workers=2))
        assert loader.persistent_workers is False

    def test_num_workers_positive_applies_timeout_by_default(self):
        loader = get_rest_dataloader(job_id=1, settings=_make_settings(num_workers=1))
        assert loader.timeout == 300


class TestDataLoaderHygieneOverrides(TestCase):
    """Operators can override or disable each knob via settings."""

    def test_mp_context_can_be_set_to_forkserver(self):
        loader = get_rest_dataloader(
            job_id=1,
            settings=_make_settings(
                num_workers=1, antenna_api_dataloader_mp_context="forkserver"
            ),
        )
        assert loader.multiprocessing_context.get_start_method() == "forkserver"

    def test_mp_context_can_be_overridden_to_spawn(self):
        loader = get_rest_dataloader(
            job_id=1,
            settings=_make_settings(
                num_workers=1, antenna_api_dataloader_mp_context="spawn"
            ),
        )
        assert loader.multiprocessing_context.get_start_method() == "spawn"

    def test_mp_context_empty_string_falls_back_to_pytorch_default(self):
        """Empty string = let PyTorch pick (the historical pre-fix behavior).

        Operators who need the old `fork` behavior on a specific host can
        set this without a code change.
        """
        loader = get_rest_dataloader(
            job_id=1,
            settings=_make_settings(
                num_workers=1, antenna_api_dataloader_mp_context=""
            ),
        )
        assert loader.multiprocessing_context is None

    def test_timeout_zero_disables_the_guard(self):
        loader = get_rest_dataloader(
            job_id=1,
            settings=_make_settings(num_workers=1, antenna_api_dataloader_timeout_s=0),
        )
        assert loader.timeout == 0

    def test_invalid_mp_context_raises(self):
        """Typoed values get caught up front instead of producing a confusing
        torch error later."""
        with pytest.raises(ValueError, match="antenna_api_dataloader_mp_context"):
            get_rest_dataloader(
                job_id=1,
                settings=_make_settings(
                    num_workers=1, antenna_api_dataloader_mp_context="not-a-real-method"
                ),
            )


class TestDataLoaderHygieneBackwardsCompat(TestCase):
    """A Settings object without the new fields must still work.

    Older deploys / test fixtures may not have the new fields yet; we use
    getattr() with sensible defaults so the worker can keep running.
    """

    def test_missing_fields_use_defaults(self):
        bare = SimpleNamespace(
            antenna_api_base_url="http://testserver/api/v2",
            antenna_api_auth_token="test-token",
            antenna_api_batch_size=2,
            num_workers=1,
        )
        loader = get_rest_dataloader(job_id=1, settings=bare)
        # Defaults: PyTorch's start method + 300s timeout
        assert loader.multiprocessing_context is None
        assert loader.timeout == 300
