"""Install-path tests for the OpenAI adapter (pip install chatcortex[real]).

These exercise what the stubbed-client tests in
``test_executor_backend.py`` cannot: the lazy ``import openai`` branch,
the messages a ``[real]`` user meets before any network call, and the
real SDK client/call objects (never the network — the call target is an
unroutable loopback port, so nothing leaves the machine).

Tests needing the SDK skip when ``openai`` is not installed (``[real]``
is an optional extra; it must not become a hard dev dependency).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from chatcortex import AgentGraph, ComponentMetadata
from chatcortex.adapters.openai_backend import OpenAIBackend
from chatcortex.execution.executor import AgentExecutor, ProbeResult

# Both are real ``openai`` transport exception type names; which one a
# given box surfaces for a dead endpoint depends on its firewall
# (connection refused -> APIConnectionError, dropped -> APITimeoutError).
TRANSPORT_ERROR_PATTERN = re.compile(r"API(Connection|Timeout)Error")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _chain():
    graph = AgentGraph()
    graph.chain_append(
        "a",
        ComponentMetadata(
            name="a",
            component_type="tool",
            capabilities=["retrieval"],
            cost_per_call=0.001,
            avg_latency_ms=10,
            reliability_score=0.5,
            privacy_level="internal",
        ),
    )
    return graph


# --- (a) lazy import branch -------------------------------------------


def test_package_import_never_touches_openai():
    """Fresh interpreter: importing chatcortex must not load the SDK."""
    pytest.importorskip("openai")
    code = (
        "import sys; import chatcortex; import chatcortex.execution; "
        "assert not [m for m in sys.modules if m.split('.')[0] == 'openai']; "
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("clean")


# --- SDK-missing path -------------------------------------------------


def test_construction_without_sdk_raises_with_install_hint():
    """No openai installed: construction explains the optional extra."""
    import importlib.util

    if importlib.util.find_spec("openai") is not None:
        pytest.skip("openai is installed; the missing-SDK path is unreachable")

    with pytest.raises(ImportError, match=r"chatcortex\[real\]"):
        OpenAIBackend()


# --- (b) missing-key path ---------------------------------------------


def test_construction_without_key_raises_openai_error(monkeypatch):
    openai = pytest.importorskip("openai")
    for key in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(openai.OpenAIError):
        OpenAIBackend()


# --- (c) dummy-key construction + real SDK objects, offline -----------


def test_construction_with_dummy_key_makes_no_request_at_init(monkeypatch):
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-test")
    backend = OpenAIBackend()

    assert isinstance(backend._client, openai.OpenAI)
    # Pure configuration: the key is attached and no request can have run.
    assert backend._client.api_key == "sk-dummy-test"
    assert str(backend._client.base_url).startswith("https://api.openai.com")


def test_real_sdk_call_path_offline_is_a_failed_probe():
    """The real SDK call path, with a client that cannot leave localhost."""
    openai = pytest.importorskip("openai")
    # Port 9 (discard protocol) typically refuses immediately; timeout
    # bounds the worst case. No external host is contacted.
    offline_client = openai.OpenAI(
        api_key="sk-dummy-test",
        base_url="http://127.0.0.1:9",
        timeout=2.0,
        max_retries=0,
    )
    backend = OpenAIBackend(model="dummy", client=offline_client)

    probe = backend(_chain().get_metadata("a"), None)

    assert isinstance(probe, ProbeResult)
    assert probe.success is False
    assert probe.cost == 0.0
    assert probe.latency_ms >= 0.0
    assert TRANSPORT_ERROR_PATTERN.search(str(probe.output))


def test_executor_records_real_sdk_failure_in_telemetry():
    """The executor's exception-probe path with a genuine SDK error."""
    openai = pytest.importorskip("openai")
    offline_client = openai.OpenAI(
        api_key="sk-dummy-test",
        base_url="http://127.0.0.1:9",
        timeout=2.0,
        max_retries=0,
    )
    backend = OpenAIBackend(model="dummy", client=offline_client)

    result = AgentExecutor(backend=backend).execute(_chain())

    records = result.telemetry.records
    assert len(records) == 1
    assert records[0]["success"] is False
    assert records[0]["cost"] == 0.0
    assert result.summary()["success"] is False
