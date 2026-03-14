from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


MODULES_TO_RELOAD = [
    "utils.store",
    "utils.metrics",
    "utils.reference_formatter",
    "utils.ai_ops",
    "utils.suggestion_service",
    "utils.chat_service",
    "utils.edit_service",
    "utils.library_service",
    "main",
]


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("CITEFLOW_DB_PATH", str(tmp_path / "citeflow.db"))
    monkeypatch.setenv("CITEFLOW_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = {}
    for module_name in MODULES_TO_RELOAD:
        if module_name in sys.modules:
            loaded[module_name] = importlib.reload(sys.modules[module_name])
        else:
            loaded[module_name] = importlib.import_module(module_name)

    app_module = loaded["main"]
    with TestClient(app_module.app) as client:
        yield client, loaded
