"""The fake engine by default; the real local engine when MERIDIAN_CONTRACT=1 (deferred by the owner)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from .contract_guard import refusal, runtime_dir
from .fake_engine import FakeEngine

CONTRACT = os.environ.get('MERIDIAN_CONTRACT') == '1'


class ContractEngine:
    is_fake = False
    transport: httpx.BaseTransport | None = None
    requests: list[httpx.Request] = []  # noqa: RUF012

    def __init__(self, requested_by: str) -> None:
        self.requested_by = requested_by  # the test node that asked, for the record

        # [1] Even a direct `MERIDIAN_CONTRACT=1 pytest` meets the guard that scripts/contract.py runs
        reason = refusal(os.environ)
        folder = os.environ.get('MERIDIAN_RUNTIME_DIR') or runtime_dir(os.environ)
        if reason is not None or folder is None:
            raise RuntimeError(f'Contract run refused: {reason or "the local runtime folder cannot be located."}')

        # [2] The URLs the guard accepted and the keys the local runtime holds
        runtime = Path(folder)
        self.public_url = os.environ.get('MERIDIAN_PUBLIC_URL', 'http://127.0.0.1:8180')
        self.admin_url = os.environ.get('MERIDIAN_ADMIN_URL', 'http://127.0.0.1:8181')
        self.project_key = (runtime / 'local-api-key.txt').read_text(encoding='utf-8').strip()
        self.integrator_key = (runtime / 'local-integrator-key.txt').read_text(encoding='utf-8').strip()

    def next(self, *_: Any) -> None:
        raise RuntimeError('scripted answers need the fake engine')


def make_engine(node: Any) -> FakeEngine | ContractEngine:
    """The engine for one test. A test that did not say which engine it is for (`fake_only` or `contract`) gets
    neither, however it reached this fixture; collection-time checks cannot see `request.getfixturevalue`."""
    if not any(node.get_closest_marker(name) for name in ('fake_only', 'contract')):
        pytest.fail(
            f'{node.nodeid} uses the engine without @pytest.mark.fake_only or @pytest.mark.contract', pytrace=False
        )
    return ContractEngine(node.nodeid) if CONTRACT else FakeEngine()
