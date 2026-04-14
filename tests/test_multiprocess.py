"""Tests for multi-process, server, and unrestricted mode modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_config_multiprocess_defaults():
    from tqcli.config import MultiProcessConfig, TqConfig
    config = TqConfig()
    assert config.multiprocess.server_port == 8741
    assert config.multiprocess.server_host == "127.0.0.1"
    assert config.multiprocess.max_workers == 3
    assert config.unrestricted is False


def test_config_multiprocess_save_load():
    from tqcli.config import TqConfig
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.yaml"
        config = TqConfig()
        config.multiprocess.max_workers = 5
        config.multiprocess.server_port = 9999
        config.unrestricted = True
        config.save(path)

        loaded = TqConfig.load(path)
        assert loaded.multiprocess.max_workers == 5
        assert loaded.multiprocess.server_port == 9999
        assert loaded.unrestricted is True


def test_server_config():
    from tqcli.core.server import ServerConfig
    cfg = ServerConfig(engine="llama.cpp", model_path="/tmp/model.gguf")
    assert cfg.port == 8741
    assert cfg.host == "127.0.0.1"
    assert cfg.engine == "llama.cpp"


def test_server_status_not_running():
    from tqcli.core.server import InferenceServer, ServerConfig
    server = InferenceServer(ServerConfig(engine="llama.cpp", model_path="/tmp/model.gguf"))
    status = server.status()
    assert status.running is False
    assert status.pid is None


def test_server_build_llama_cmd():
    from tqcli.core.server import InferenceServer, ServerConfig
    cfg = ServerConfig(
        engine="llama.cpp", model_path="/tmp/model.gguf",
        port=8741, context_length=4096, n_gpu_layers=-1,
    )
    server = InferenceServer(cfg)
    cmd = server._build_llama_cmd()
    assert "--model" in cmd
    assert "/tmp/model.gguf" in cmd
    assert "--port" in cmd
    assert "8741" in cmd


def test_server_build_vllm_cmd():
    from tqcli.core.server import InferenceServer, ServerConfig
    cfg = ServerConfig(
        engine="vllm", model_path="/tmp/model",
        port=8741, context_length=8192, quantization="awq",
    )
    server = InferenceServer(cfg)
    cmd = server._build_vllm_cmd()
    assert "--model" in cmd
    assert "--quantization" in cmd
    assert "awq" in cmd


def test_estimate_server_resources_llamacpp():
    from tqcli.core.server import estimate_server_resources
    est = estimate_server_resources(
        engine="llama.cpp", model_size_mb=4500, n_workers=3,
        vram_mb=4096, ram_mb=32000,
    )
    assert est["engine"] == "llama.cpp"
    assert est["fits"] is True
    assert est["model_mb"] == 4500
    assert "sequential" in est["note"].lower()


def test_estimate_server_resources_vllm():
    from tqcli.core.server import estimate_server_resources
    est = estimate_server_resources(
        engine="vllm", model_size_mb=4500, n_workers=3,
        vram_mb=24000, ram_mb=32000,
    )
    assert est["engine"] == "vllm"
    assert est["fits"] is True
    assert "PagedAttention" in est["note"]


def test_estimate_server_resources_vllm_insufficient():
    from tqcli.core.server import estimate_server_resources
    est = estimate_server_resources(
        engine="vllm", model_size_mb=20000, n_workers=5,
        vram_mb=4096, ram_mb=32000,
    )
    assert est["fits"] is False


def test_assess_multiprocess_feasible():
    from tqcli.core.multiprocess import assess_multiprocess
    from tqcli.core.system_info import SystemInfo

    sys_info = SystemInfo(
        os_name="linux", os_version="6.0", os_display="Linux",
        arch="x86_64", cpu_cores_physical=8, cpu_cores_logical=16,
        ram_total_mb=32000, ram_available_mb=28000,
        llama_cpp_available=True,
    )
    plan = assess_multiprocess(
        sys_info=sys_info,
        model_path="/tmp/model.gguf",
        model_size_mb=4500,
        requested_workers=3,
        preferred_engine="llama.cpp",
    )
    assert plan.feasible is True
    assert plan.engine == "llama.cpp"
    assert plan.max_workers >= 1


def test_assess_multiprocess_unrestricted_overrides():
    from tqcli.core.multiprocess import assess_multiprocess
    from tqcli.core.system_info import SystemInfo

    sys_info = SystemInfo(
        os_name="linux", os_version="6.0", os_display="Linux",
        arch="x86_64", cpu_cores_physical=2, cpu_cores_logical=4,
        ram_total_mb=4000, ram_available_mb=2000,
        llama_cpp_available=True,
    )
    # Without unrestricted: may not be feasible
    plan_normal = assess_multiprocess(
        sys_info=sys_info, model_path="/tmp/model.gguf",
        model_size_mb=8000, requested_workers=3,
    )

    # With unrestricted: forces feasible
    plan_unrestricted = assess_multiprocess(
        sys_info=sys_info, model_path="/tmp/model.gguf",
        model_size_mb=8000, requested_workers=3, unrestricted=True,
    )
    assert plan_unrestricted.feasible is True
    assert any("unrestricted" in w.lower() or "stop-trying" in w.lower() or "proceeding" in w.lower()
               for w in plan_unrestricted.warnings)


def test_server_client_backend_properties():
    from tqcli.core.server_client import ServerClientBackend
    client = ServerClientBackend(base_url="http://127.0.0.1:8741", model_name="test")
    assert client.engine_name == "server-client"
    assert client.is_available is True
    assert client.is_loaded is False


def test_unrestricted_mode_flag():
    from tqcli.core.unrestricted import is_unrestricted

    class FakeCtx:
        obj = {"unrestricted": True}
    assert is_unrestricted(FakeCtx()) is True

    class FakeCtx2:
        obj = {"unrestricted": False}
    assert is_unrestricted(FakeCtx2()) is False


def test_unrestricted_warning_text():
    from tqcli.core.unrestricted import _UNRESTRICTED_WARNING
    assert "stop-trying-to-control-everything-and-just-let-go" in _UNRESTRICTED_WARNING
    assert "BYPASSED" in _UNRESTRICTED_WARNING
    assert "STILL ON" in _UNRESTRICTED_WARNING


def test_skill_loader_finds_multiprocess():
    from tqcli.skills.loader import SkillLoader
    skills_dir = Path(__file__).parent.parent / ".claude" / "skills"
    if skills_dir.exists():
        loader = SkillLoader([skills_dir])
        skills = loader.get_tq_skills()
        names = [s.name for s in skills]
        assert "tq-multi-process" in names


def test_cli_unrestricted_flag():
    """Test that the CLI accepts the unrestricted flag."""
    from click.testing import CliRunner
    from tqcli.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--stop-trying-to-control-everything-and-just-let-go", "--version"])
    assert "0.3.1" in result.output


def test_cli_serve_status():
    """Test serve status when no server is running."""
    from click.testing import CliRunner
    from tqcli.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["serve", "status"])
    assert "No inference server running" in result.output


def test_cli_workers_list():
    """Test workers list when no workers are running."""
    from click.testing import CliRunner
    from tqcli.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["workers", "list"])
    assert "No active workers" in result.output or "worker" in result.output.lower()
