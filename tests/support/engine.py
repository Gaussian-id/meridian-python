"""The fake engine by default; the real local engine when MERIDIAN_CONTRACT=1 (deferred by the owner)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from .contract_guard import refusal, runtime_dir
from .fake_engine import FakeEngine

CONTRACT = os.environ.get('MERIDIAN_CONTRACT') == '1'


class ContractEngine:
    is_fake = False
    transport: httpx.BaseTransport | None = None
    requests: list[httpx.Request] = []  # noqa: RUF012

    def __init__(self) -> None:
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


def make_engine() -> FakeEngine | ContractEngine:
    return ContractEngine() if CONTRACT else FakeEngine()
