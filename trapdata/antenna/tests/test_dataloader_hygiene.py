"""Regression test for DataLoader subprocess-hygiene settings.

This test guards the fix shipped for ami-data-companion#140 / #145:
``get_rest_dataloader()`` must apply the ``multiprocessing_context`` and
``timeout`` knobs from settings. Without these, the DataLoader inherits the
parent's heap via ``fork`` (leaking CUDA / pinned-memory state into shared
memory) and silently hangs forever when a subprocess dies mid-batch.

The RSS-growth side of the regression is already covered by
``test_memory_leak.py``; this file focuses on the *configuration* surface so
the fix can't be silently regressed by a future refactor of
``get_rest_dataloader``.
"""

import multiprocessing
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
    defaults = dict(
        antenna_api_base_url="http://testserver/api/v2",
        antenna_api_auth_token="test-token",
        antenna_api_batch_size=2,
        num_workers=0,
        antenna_api_dataloader_mp_context="forkserver",
        antenna_api_dataloader_timeout_s=300,
    )
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

    def test_num_workers_positive_applies_forkserver_context_by_default(self):
        """num_workers > 0 = mp_context must be the forkserver context."""
        loader = get_rest_dataloader(job_id=1, settings=_make_settings(num_workers=1))
        ctx = loader.multiprocessing_context
        assert ctx is not None, "DataLoader must have an explicit multiprocessing context"
        # multiprocessing.get_context returns one of the context singletons; check
        # its start method matches what we configured.
        assert ctx.get_start_method() == "forkserver"

    def test_num_workers_positive_applies_timeout_by_default(self):
        loader = get_rest_dataloader(job_id=1, settings=_make_settings(num_workers=1))
        assert loader.timeout == 300


class TestDataLoaderHygieneOverrides(TestCase):
    """Operators can override or disable each knob via settings."""

    def test_mp_context_can_be_overridden_to_spawn(self):
        loader = get_rest_dataloader(
            job_id=1,
            settings=_make_settings(num_workers=1, antenna_api_dataloader_mp_context="spawn"),
        )
        assert loader.multiprocessing_context.get_start_method() == "spawn"

    def test_mp_context_empty_string_falls_back_to_pytorch_default(self):
        """Empty string = let PyTorch pick (the historical pre-fix behavior).

        Operators who need the old `fork` behavior on a specific host can
        set this without a code change.
        """
        loader = get_rest_dataloader(
            job_id=1,
            settings=_make_settings(num_workers=1, antenna_api_dataloader_mp_context=""),
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
        # Defaults: forkserver context + 300s timeout
        assert loader.multiprocessing_context is not None
        assert loader.multiprocessing_context.get_start_method() == "forkserver"
        assert loader.timeout == 300
