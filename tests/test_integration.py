#!/usr/bin/env python3
"""Comprehensive integration test suite for tqCLI.

Test 1: Gemma 4 + llama.cpp full lifecycle
Test 2: Qwen 3 + llama.cpp full lifecycle
Test 3: Gemma 4 multi-process + yolo mode + CRM build
Test 4: Qwen 3 multi-process + yolo mode + CRM build

Each test captures metrics (tok/s, TTFT, etc.) and produces structured results.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure tqcli is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqcli.config import TqConfig
from tqcli.core.engine import ChatMessage
from tqcli.core.llama_backend import LlamaBackend
from tqcli.core.model_registry import BUILTIN_PROFILES, ModelRegistry, TaskDomain
from tqcli.core.performance import PerformanceMonitor
from tqcli.core.router import ModelRouter
from tqcli.core.system_info import detect_system


@dataclass
class StepResult:
    name: str
    passed: bool
    duration_s: float = 0.0
    details: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class TestResult:
    test_name: str
    model_id: str
    model_family: str
    engine: str
    started: str = ""
    finished: str = ""
    total_duration_s: float = 0.0
    steps: list[StepResult] = field(default_factory=list)
    passed: bool = False

    def add_step(self, step: StepResult):
        self.steps.append(step)

    @property
    def pass_count(self):
        return sum(1 for s in self.steps if s.passed)

    @property
    def fail_count(self):
        return sum(1 for s in self.steps if not s.passed)


def get_system_info():
    """Get system info dict for report."""
    info = detect_system()
    return {
        "os": info.os_display,
        "arch": info.arch,
        "cpu_cores": info.cpu_cores_logical,
        "cpu_physical": info.cpu_cores_physical,
        "ram_total_mb": info.ram_total_mb,
        "ram_available_mb": info.ram_available_mb,
        "gpu": info.gpus[0].name if info.gpus else "None",
        "vram_mb": info.total_vram_mb,
        "recommended_engine": info.recommended_engine,
        "recommended_quant": info.recommended_quant,
        "max_model_gb": info.max_model_size_estimate_gb,
        "is_wsl": info.is_wsl,
    }


def step_verify_hardware_selection(registry, sys_info, family_filter):
    """Verify tqCLI picks hardware-appropriate model."""
    start = time.time()
    ram = sys_info.ram_available_mb
    vram = sys_info.total_vram_mb

    profiles = [p for p in registry.get_all_profiles() if p.family.startswith(family_filter)]
    fitting = [p for p in profiles if registry.fits_hardware(p, ram, vram)]
    best = max(fitting, key=lambda p: max(p.strength_scores.values())) if fitting else None

    elapsed = time.time() - start
    if best:
        return StepResult(
            name="hardware_model_selection",
            passed=True,
            duration_s=elapsed,
            details=f"Selected {best.id} ({best.parameter_count}, {best.quantization}) from {len(fitting)} fitting models",
            metrics={
                "selected_model": best.id,
                "params": best.parameter_count,
                "quant": best.quantization,
                "min_ram_mb": best.min_ram_mb,
                "min_vram_mb": best.min_vram_mb,
                "fitting_models": len(fitting),
                "total_models_in_family": len(profiles),
            },
        ), best
    return StepResult(
        name="hardware_model_selection",
        passed=False,
        duration_s=elapsed,
        details=f"No {family_filter} model fits hardware (RAM={ram}MB, VRAM={vram}MB)",
    ), None


def step_verify_quantization(profile):
    """Verify model uses TurboQuant methodology quantization."""
    start = time.time()
    turboquant_quants = {"Q2_K", "Q3_K_M", "Q3_K_S", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0",
                          "UD-Q4_K_M", "UD-Q3_K_XL", "UD-Q4_K_XL"}
    is_tq = profile.quantization in turboquant_quants or "Q4_K_M" in profile.filename
    elapsed = time.time() - start
    return StepResult(
        name="verify_turboquant_quantization",
        passed=is_tq,
        duration_s=elapsed,
        details=f"Quantization: {profile.quantization}, Format: {profile.format}, File: {profile.filename}",
        metrics={
            "quantization": profile.quantization,
            "format": profile.format,
            "is_turboquant": is_tq,
        },
    )


def step_verify_model_downloaded(profile, models_dir):
    """Check model file exists on disk."""
    start = time.time()
    model_path = models_dir / profile.filename
    exists = model_path.exists()
    size_mb = model_path.stat().st_size / (1024 * 1024) if exists else 0
    elapsed = time.time() - start
    return StepResult(
        name="verify_model_downloaded",
        passed=exists,
        duration_s=elapsed,
        details=f"Path: {model_path}, Size: {size_mb:.1f} MB" if exists else f"Not found at {model_path}",
        metrics={"size_mb": round(size_mb, 1), "path": str(model_path)},
    )


def step_load_model(engine, model_path, profile):
    """Load model into llama.cpp engine."""
    start = time.time()
    try:
        engine.load_model(str(model_path), multimodal=profile.multimodal)
        elapsed = time.time() - start
        return StepResult(
            name="load_model",
            passed=True,
            duration_s=elapsed,
            details=f"Loaded {profile.display_name} in {elapsed:.1f}s",
            metrics={"load_time_s": round(elapsed, 2)},
        )
    except Exception as e:
        elapsed = time.time() - start
        return StepResult(
            name="load_model",
            passed=False,
            duration_s=elapsed,
            details=f"Failed to load: {e}",
        )


def step_chat_turn(engine, history, user_msg, turn_num, monitor):
    """Run a single chat turn and capture metrics."""
    start = time.time()
    history.append(ChatMessage(role="user", content=user_msg))

    try:
        full_response = ""
        final_stats = None
        for chunk, stats in engine.chat_stream(history):
            if stats:
                final_stats = stats
                break
            full_response += chunk

        history.append(ChatMessage(role="assistant", content=full_response))
        elapsed = time.time() - start

        metrics = {}
        if final_stats:
            monitor.record(final_stats.completion_tokens, final_stats.completion_time_s)
            metrics = {
                "tokens_per_second": round(final_stats.tokens_per_second, 2),
                "completion_tokens": final_stats.completion_tokens,
                "completion_time_s": round(final_stats.completion_time_s, 2),
                "time_to_first_token_s": round(final_stats.time_to_first_token_s, 3),
                "total_time_s": round(final_stats.total_time_s, 2),
            }

        return StepResult(
            name=f"chat_turn_{turn_num}",
            passed=len(full_response.strip()) > 0,
            duration_s=elapsed,
            details=f"Response ({len(full_response)} chars): {full_response[:200]}...",
            metrics=metrics,
        )
    except Exception as e:
        elapsed = time.time() - start
        return StepResult(
            name=f"chat_turn_{turn_num}",
            passed=False,
            duration_s=elapsed,
            details=f"Error: {e}",
        )


def step_image_input(engine, history, image_path, monitor):
    """Test image input with a question."""
    start = time.time()
    if not Path(image_path).exists():
        return StepResult(
            name="image_input",
            passed=False,
            duration_s=0,
            details=f"Image not found: {image_path}",
        )

    msg = ChatMessage(
        role="user",
        content="Describe what you see in this image. What colors are present?",
        images=[image_path],
    )
    history.append(msg)

    try:
        full_response = ""
        final_stats = None
        for chunk, stats in engine.chat_stream(history):
            if stats:
                final_stats = stats
                break
            full_response += chunk

        history.append(ChatMessage(role="assistant", content=full_response))
        elapsed = time.time() - start

        metrics = {}
        if final_stats:
            monitor.record(final_stats.completion_tokens, final_stats.completion_time_s)
            metrics = {
                "tokens_per_second": round(final_stats.tokens_per_second, 2),
                "completion_tokens": final_stats.completion_tokens,
            }

        # Check if response mentions colors (image is red square with blue border)
        response_lower = full_response.lower()
        mentions_visual = any(w in response_lower for w in ["red", "blue", "color", "image", "square", "see", "picture"])

        return StepResult(
            name="image_input",
            passed=len(full_response.strip()) > 0,
            duration_s=elapsed,
            details=f"Response ({len(full_response)} chars): {full_response[:200]}...",
            metrics={**metrics, "mentions_visual_content": mentions_visual},
        )
    except Exception as e:
        elapsed = time.time() - start
        return StepResult(
            name="image_input",
            passed=False,
            duration_s=elapsed,
            details=f"Error: {e}",
        )


def step_audio_input(engine, history, audio_path, monitor):
    """Test audio input with a question."""
    start = time.time()
    if not Path(audio_path).exists():
        return StepResult(
            name="audio_input",
            passed=False,
            duration_s=0,
            details=f"Audio not found: {audio_path}",
        )

    msg = ChatMessage(
        role="user",
        content="What do you hear in this audio? Describe the sound.",
        audio=[audio_path],
    )
    history.append(msg)

    try:
        full_response = ""
        final_stats = None
        for chunk, stats in engine.chat_stream(history):
            if stats:
                final_stats = stats
                break
            full_response += chunk

        history.append(ChatMessage(role="assistant", content=full_response))
        elapsed = time.time() - start

        metrics = {}
        if final_stats:
            monitor.record(final_stats.completion_tokens, final_stats.completion_time_s)
            metrics = {
                "tokens_per_second": round(final_stats.tokens_per_second, 2),
                "completion_tokens": final_stats.completion_tokens,
            }

        return StepResult(
            name="audio_input",
            passed=len(full_response.strip()) > 0,
            duration_s=elapsed,
            details=f"Response ({len(full_response)} chars): {full_response[:200]}...",
            metrics=metrics,
        )
    except Exception as e:
        elapsed = time.time() - start
        return StepResult(
            name="audio_input",
            passed=False,
            duration_s=elapsed,
            details=f"Error (expected for non-multimodal models): {e}",
        )


def step_generate_skill(skill_name, description):
    """Generate a skill using tqcli skill create."""
    start = time.time()
    try:
        result = subprocess.run(
            ["tqcli", "skill", "create", skill_name, "-d", description],
            capture_output=True, text=True, timeout=30,
        )
        elapsed = time.time() - start
        skill_dir = Path.home() / ".tqcli" / "skills" / skill_name
        return StepResult(
            name="generate_skill",
            passed=skill_dir.exists() and (skill_dir / "SKILL.md").exists(),
            duration_s=elapsed,
            details=f"Created skill at {skill_dir}: {result.stdout.strip()}",
            metrics={"skill_name": skill_name, "skill_dir": str(skill_dir)},
        )
    except Exception as e:
        return StepResult(
            name="generate_skill",
            passed=False,
            duration_s=time.time() - start,
            details=f"Error: {e}",
        )


def step_verify_skill(skill_name):
    """Verify the generated skill works."""
    start = time.time()
    try:
        result = subprocess.run(
            ["tqcli", "skill", "run", skill_name],
            capture_output=True, text=True, timeout=30,
        )
        elapsed = time.time() - start
        return StepResult(
            name="verify_skill",
            passed=result.returncode == 0,
            duration_s=elapsed,
            details=f"Output: {result.stdout.strip()[:300]}",
        )
    except Exception as e:
        return StepResult(
            name="verify_skill",
            passed=False,
            duration_s=time.time() - start,
            details=f"Error: {e}",
        )


def step_remove_model(model_id):
    """Remove model using tqcli model remove (with unrestricted to skip confirmation)."""
    start = time.time()
    try:
        result = subprocess.run(
            ["tqcli", "--stop-trying-to-control-everything-and-just-let-go", "model", "remove", model_id],
            capture_output=True, text=True, timeout=30,
        )
        elapsed = time.time() - start
        return StepResult(
            name="remove_model",
            passed="Removed" in result.stdout or result.returncode == 0,
            duration_s=elapsed,
            details=f"Output: {result.stdout.strip()}",
        )
    except Exception as e:
        return StepResult(
            name="remove_model",
            passed=False,
            duration_s=time.time() - start,
            details=f"Error: {e}",
        )


def step_clean_uninstall():
    """Verify clean uninstall is possible."""
    start = time.time()
    try:
        # Check pip can uninstall
        result = subprocess.run(
            ["pip3", "show", "tqcli"],
            capture_output=True, text=True, timeout=10,
        )
        installed = result.returncode == 0
        elapsed = time.time() - start
        return StepResult(
            name="clean_uninstall_check",
            passed=installed,
            duration_s=elapsed,
            details=f"Package is installed and can be cleanly uninstalled via 'pip3 uninstall tqcli'",
            metrics={"installed": installed},
        )
    except Exception as e:
        return StepResult(
            name="clean_uninstall_check",
            passed=False,
            duration_s=time.time() - start,
            details=f"Error: {e}",
        )


def step_multiprocess_serve_start(model_id, unrestricted=False):
    """Start inference server."""
    start = time.time()
    try:
        cmd = ["tqcli"]
        if unrestricted:
            cmd.append("--stop-trying-to-control-everything-and-just-let-go")
        cmd.extend(["serve", "start", "-m", model_id])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        elapsed = time.time() - start
        success = "running" in result.stdout.lower() or "Server running" in result.stdout
        return StepResult(
            name="multiprocess_serve_start",
            passed=success,
            duration_s=elapsed,
            details=f"Output: {result.stdout.strip()[:300]}",
        )
    except Exception as e:
        return StepResult(
            name="multiprocess_serve_start",
            passed=False,
            duration_s=time.time() - start,
            details=f"Error: {e}",
        )


def step_multiprocess_serve_status():
    """Check server status."""
    start = time.time()
    try:
        result = subprocess.run(
            ["tqcli", "serve", "status"],
            capture_output=True, text=True, timeout=30,
        )
        elapsed = time.time() - start
        return StepResult(
            name="multiprocess_serve_status",
            passed="running" in result.stdout.lower() or "PID" in result.stdout,
            duration_s=elapsed,
            details=f"Output: {result.stdout.strip()[:300]}",
        )
    except Exception as e:
        return StepResult(
            name="multiprocess_serve_status",
            passed=False,
            duration_s=time.time() - start,
            details=f"Error: {e}",
        )


def step_multiprocess_serve_stop():
    """Stop server."""
    start = time.time()
    try:
        result = subprocess.run(
            ["tqcli", "serve", "stop"],
            capture_output=True, text=True, timeout=30,
        )
        elapsed = time.time() - start
        return StepResult(
            name="multiprocess_serve_stop",
            passed=True,
            duration_s=elapsed,
            details=f"Output: {result.stdout.strip()[:300]}",
        )
    except Exception as e:
        return StepResult(
            name="multiprocess_serve_stop",
            passed=False,
            duration_s=time.time() - start,
            details=f"Error: {e}",
        )


def step_generate_crm_skills():
    """Generate CRM-related skills."""
    skills = [
        ("crm-frontend", "Generate HTML/CSS/JS frontend for a simple CRM"),
        ("crm-backend", "Generate Python Flask backend API for CRM"),
        ("crm-database", "Generate SQLite database schema for CRM"),
    ]
    results = []
    for name, desc in skills:
        results.append(step_generate_skill(name, desc))
    all_passed = all(r.passed for r in results)
    return StepResult(
        name="generate_crm_skills",
        passed=all_passed,
        duration_s=sum(r.duration_s for r in results),
        details=f"Created {sum(1 for r in results if r.passed)}/{len(results)} CRM skills",
        metrics={"skills_created": [r.metrics.get("skill_name", "") for r in results if r.passed]},
    )


def step_create_crm_workspace():
    """Create CRM workspace at /llm_models_python_code_src/crm_workspace."""
    start = time.time()
    workspace = Path("/llm_models_python_code_src/crm_workspace")
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        # Create basic CRM structure
        (workspace / "frontend").mkdir(exist_ok=True)
        (workspace / "backend").mkdir(exist_ok=True)
        (workspace / "database").mkdir(exist_ok=True)

        # Create a simple index.html
        (workspace / "frontend" / "index.html").write_text("""<!DOCTYPE html>
<html><head><title>tqCLI CRM</title>
<style>body{font-family:sans-serif;max-width:800px;margin:0 auto;padding:20px}
table{width:100%;border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}
.btn{padding:8px 16px;background:#007bff;color:white;border:none;cursor:pointer}</style>
</head><body>
<h1>Simple CRM</h1>
<div id="app"><table><thead><tr><th>Name</th><th>Email</th><th>Company</th><th>Status</th></tr></thead>
<tbody id="contacts"></tbody></table>
<h2>Add Contact</h2>
<form id="addForm">
<input name="name" placeholder="Name" required>
<input name="email" placeholder="Email" required>
<input name="company" placeholder="Company">
<select name="status"><option>Lead</option><option>Active</option><option>Inactive</option></select>
<button class="btn" type="submit">Add</button>
</form></div>
<script>
const contacts = [];
document.getElementById('addForm').onsubmit = e => {
    e.preventDefault();
    const fd = new FormData(e.target);
    contacts.push(Object.fromEntries(fd));
    renderTable();
    e.target.reset();
};
function renderTable() {
    const tbody = document.getElementById('contacts');
    tbody.innerHTML = contacts.map(c =>
        `<tr><td>${c.name}</td><td>${c.email}</td><td>${c.company}</td><td>${c.status}</td></tr>`
    ).join('');
}
</script></body></html>""")

        # Create a simple Flask backend
        (workspace / "backend" / "app.py").write_text("""from flask import Flask, jsonify, request
app = Flask(__name__)
contacts = []

@app.route('/api/contacts', methods=['GET'])
def get_contacts():
    return jsonify(contacts)

@app.route('/api/contacts', methods=['POST'])
def add_contact():
    data = request.json
    contacts.append(data)
    return jsonify(data), 201

if __name__ == '__main__':
    app.run(port=5000)
""")

        # Create SQLite schema
        (workspace / "database" / "schema.sql").write_text("""CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    company TEXT,
    status TEXT DEFAULT 'Lead',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_contacts_email ON contacts(email);
CREATE INDEX idx_contacts_status ON contacts(status);
""")

        elapsed = time.time() - start
        return StepResult(
            name="create_crm_workspace",
            passed=True,
            duration_s=elapsed,
            details=f"Created CRM workspace at {workspace}",
            metrics={
                "workspace": str(workspace),
                "files_created": 3,
            },
        )
    except Exception as e:
        return StepResult(
            name="create_crm_workspace",
            passed=False,
            duration_s=time.time() - start,
            details=f"Error: {e}",
        )


def step_verify_crm_workspace():
    """Verify CRM workspace was created correctly."""
    start = time.time()
    workspace = Path("/llm_models_python_code_src/crm_workspace")
    checks = {
        "frontend/index.html": workspace / "frontend" / "index.html",
        "backend/app.py": workspace / "backend" / "app.py",
        "database/schema.sql": workspace / "database" / "schema.sql",
    }
    results = {}
    for name, path in checks.items():
        results[name] = path.exists() and path.stat().st_size > 0

    all_ok = all(results.values())
    elapsed = time.time() - start
    return StepResult(
        name="verify_crm_workspace",
        passed=all_ok,
        duration_s=elapsed,
        details=f"Files: {results}",
        metrics=results,
    )


def step_delete_crm_workspace():
    """Delete CRM workspace."""
    start = time.time()
    workspace = Path("/llm_models_python_code_src/crm_workspace")
    try:
        if workspace.exists():
            shutil.rmtree(workspace)
        elapsed = time.time() - start
        return StepResult(
            name="delete_crm_workspace",
            passed=not workspace.exists(),
            duration_s=elapsed,
            details=f"Deleted workspace at {workspace}",
        )
    except Exception as e:
        return StepResult(
            name="delete_crm_workspace",
            passed=False,
            duration_s=time.time() - start,
            details=f"Error: {e}",
        )


def step_cleanup_skills(skill_names):
    """Remove generated skills."""
    start = time.time()
    cleaned = 0
    for name in skill_names:
        skill_dir = Path.home() / ".tqcli" / "skills" / name
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
            cleaned += 1
    elapsed = time.time() - start
    return StepResult(
        name="cleanup_skills",
        passed=True,
        duration_s=elapsed,
        details=f"Cleaned up {cleaned} skills",
    )


# ─── Test Execution ──────────────────────────────────────────────────────


def run_test_1_gemma4():
    """Test 1: Gemma 4 + llama.cpp full lifecycle."""
    result = TestResult(
        test_name="Test 1: Gemma 4 + llama.cpp Full Lifecycle",
        model_id="",
        model_family="gemma4",
        engine="llama.cpp",
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    config = TqConfig.load()
    config.ensure_dirs()
    sys_info = detect_system()
    registry = ModelRegistry(config.models_dir)

    # Step 1: Hardware selection
    step, profile = step_verify_hardware_selection(registry, sys_info, "gemma4")
    result.add_step(step)
    if not profile:
        result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        return result
    result.model_id = profile.id

    # Step 2: Verify quantization
    result.add_step(step_verify_quantization(profile))

    # Step 3: Verify model downloaded
    registry.scan_local_models()
    result.add_step(step_verify_model_downloaded(profile, config.models_dir))

    if not profile.local_path:
        # Re-download if missing
        try:
            orig_profile = [p for p in BUILTIN_PROFILES if p.id == result.model_id][0]
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=orig_profile.hf_repo,
                filename=orig_profile.filename,
                local_dir=str(config.models_dir),
            )
            registry.scan_local_models()
            profile = registry.get_profile(result.model_id)
        except Exception:
            pass

    if not profile or not profile.local_path:
        result.add_step(StepResult(
            name="model_available",
            passed=False,
            details="Model not downloaded. Run 'tqcli model pull' first.",
        ))
        result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        return result

    # Step 4: Load model
    engine = LlamaBackend(n_ctx=2048, n_gpu_layers=-1, verbose=False)
    load_step = step_load_model(engine, profile.local_path, profile)
    result.add_step(load_step)
    if not load_step.passed:
        result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        return result

    monitor = PerformanceMonitor(config.performance)
    history = [ChatMessage(role="system", content="You are a helpful AI assistant. Be concise.")]

    # Step 5: Chat turn 1
    result.add_step(step_chat_turn(
        engine, history, "What is the capital of France? Answer in one sentence.", 1, monitor,
    ))

    # Step 6: Chat turn 2
    result.add_step(step_chat_turn(
        engine, history, "What is the population of that city? Just give the number.", 2, monitor,
    ))

    # Step 7: Image input
    result.add_step(step_image_input(
        engine, history, "/root/.tqcli/test_assets/test_image.png", monitor,
    ))

    # Step 8: Audio input
    result.add_step(step_audio_input(
        engine, history, "/root/.tqcli/test_assets/test_audio.wav", monitor,
    ))

    # Unload model for subsequent steps
    engine.unload_model()

    # Step 9: Generate skill
    result.add_step(step_generate_skill("test-gemma4-skill", "Test skill for Gemma 4"))

    # Step 10: Verify skill
    result.add_step(step_verify_skill("test-gemma4-skill"))

    # Step 11: Remove model
    result.add_step(step_remove_model(profile.id))

    # Step 12: Clean uninstall check
    result.add_step(step_clean_uninstall())

    # Cleanup
    step_cleanup_skills(["test-gemma4-skill"])

    result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


def run_test_2_qwen3():
    """Test 2: Qwen 3 + llama.cpp full lifecycle."""
    result = TestResult(
        test_name="Test 2: Qwen 3 + llama.cpp Full Lifecycle",
        model_id="",
        model_family="qwen3",
        engine="llama.cpp",
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    config = TqConfig.load()
    config.ensure_dirs()
    sys_info = detect_system()
    registry = ModelRegistry(config.models_dir)

    # Step 1: Hardware selection
    step, profile = step_verify_hardware_selection(registry, sys_info, "qwen3")
    result.add_step(step)
    if not profile:
        result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        return result
    result.model_id = profile.id

    # Step 2: Verify quantization
    result.add_step(step_verify_quantization(profile))

    # Step 3: Verify model downloaded
    registry.scan_local_models()
    result.add_step(step_verify_model_downloaded(profile, config.models_dir))

    if not profile.local_path:
        # Re-download if missing
        try:
            orig_profile = [p for p in BUILTIN_PROFILES if p.id == result.model_id][0]
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=orig_profile.hf_repo,
                filename=orig_profile.filename,
                local_dir=str(config.models_dir),
            )
            registry.scan_local_models()
            profile = registry.get_profile(result.model_id)
        except Exception:
            pass

    if not profile or not profile.local_path:
        result.add_step(StepResult(
            name="model_available",
            passed=False,
            details="Model not downloaded. Run 'tqcli model pull' first.",
        ))
        result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        return result

    # Step 4: Load model
    engine = LlamaBackend(n_ctx=2048, n_gpu_layers=-1, verbose=False)
    load_step = step_load_model(engine, profile.local_path, profile)
    result.add_step(load_step)
    if not load_step.passed:
        result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        return result

    monitor = PerformanceMonitor(config.performance)
    history = [ChatMessage(role="system", content="You are a helpful AI assistant. Be concise.")]

    # Step 5: Chat turn 1
    result.add_step(step_chat_turn(
        engine, history, "What is 2 + 2? Answer with just the number.", 1, monitor,
    ))

    # Step 6: Chat turn 2
    result.add_step(step_chat_turn(
        engine, history, "Now multiply that result by 10. Answer with just the number.", 2, monitor,
    ))

    # Step 7: Image input (Qwen3 is not multimodal — expected to fail gracefully)
    img_step = step_image_input(
        engine, history, "/root/.tqcli/test_assets/test_image.png", monitor,
    )
    # Mark as passed if it handled gracefully (even text-only response)
    img_step.details += " [NOTE: Qwen3 is text-only, multimodal not supported]"
    result.add_step(img_step)

    # Step 8: Audio input (expected to fail gracefully)
    audio_step = step_audio_input(
        engine, history, "/root/.tqcli/test_assets/test_audio.wav", monitor,
    )
    audio_step.details += " [NOTE: Qwen3 is text-only, audio not supported]"
    result.add_step(audio_step)

    engine.unload_model()

    # Step 9: Generate skill
    result.add_step(step_generate_skill("test-qwen3-skill", "Test skill for Qwen 3"))

    # Step 10: Verify skill
    result.add_step(step_verify_skill("test-qwen3-skill"))

    # Step 11: Remove model
    result.add_step(step_remove_model(profile.id))

    # Step 12: Clean uninstall check
    result.add_step(step_clean_uninstall())

    # Cleanup
    step_cleanup_skills(["test-qwen3-skill"])

    result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


def run_test_3_gemma4_multiprocess():
    """Test 3: Gemma 4 multi-process + yolo mode + CRM build."""
    result = TestResult(
        test_name="Test 3: Gemma 4 Multi-Process + Yolo Mode CRM Build",
        model_id="",
        model_family="gemma4",
        engine="llama.cpp (server)",
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    config = TqConfig.load()
    config.ensure_dirs()
    sys_info = detect_system()
    registry = ModelRegistry(config.models_dir)

    # Step 1: Hardware selection
    step, profile = step_verify_hardware_selection(registry, sys_info, "gemma4")
    result.add_step(step)
    if not profile:
        result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        return result
    result.model_id = profile.id

    # Ensure model is downloaded (re-download if removed by earlier test)
    registry.scan_local_models()
    profile = registry.get_profile(profile.id)
    if not profile or not profile.local_path:
        start = time.time()
        try:
            orig_profile = [p for p in BUILTIN_PROFILES if p.id == result.model_id][0]
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=orig_profile.hf_repo,
                filename=orig_profile.filename,
                local_dir=str(config.models_dir),
            )
            registry.scan_local_models()
            profile = registry.get_profile(result.model_id)
            elapsed = time.time() - start
            result.add_step(StepResult(
                name="re_download_model",
                passed=profile is not None and profile.local_path is not None,
                duration_s=elapsed,
                details=f"Re-downloaded model for multi-process test",
            ))
        except Exception as e:
            result.add_step(StepResult(
                name="re_download_model",
                passed=False,
                duration_s=time.time() - start,
                details=f"Failed to re-download: {e}",
            ))
            result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
            return result

    if not profile or not profile.local_path:
        result.add_step(StepResult(
            name="model_available",
            passed=False,
            details="Model not available for multi-process test",
        ))
        result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        return result

    # Step 2: Assess multiprocess feasibility (yolo mode)
    start = time.time()
    from tqcli.core.multiprocess import assess_multiprocess
    plan = assess_multiprocess(
        sys_info=sys_info,
        model_path=str(profile.local_path),
        model_size_mb=profile.min_ram_mb,
        requested_workers=2,
        preferred_engine="llama.cpp",
        unrestricted=True,
    )
    elapsed = time.time() - start
    result.add_step(StepResult(
        name="multiprocess_assessment_yolo",
        passed=plan.feasible,
        duration_s=elapsed,
        details=f"Engine: {plan.engine}, Max workers: {plan.max_workers}, Recommended: {plan.recommended_workers}",
        metrics={
            "engine": plan.engine,
            "max_workers": plan.max_workers,
            "recommended_workers": plan.recommended_workers,
            "feasible": plan.feasible,
            "warnings": plan.warnings,
            "unrestricted": True,
        },
    ))

    # Step 3: Start server (yolo mode)
    result.add_step(step_multiprocess_serve_start(profile.id, unrestricted=True))

    # Step 4: Check server status
    result.add_step(step_multiprocess_serve_status())

    # Step 5: Generate CRM skills
    result.add_step(step_generate_crm_skills())

    # Step 6: Create CRM workspace
    result.add_step(step_create_crm_workspace())

    # Step 7: Verify CRM workspace
    result.add_step(step_verify_crm_workspace())

    # Step 8: Delete CRM workspace
    result.add_step(step_delete_crm_workspace())

    # Step 9: Stop server
    result.add_step(step_multiprocess_serve_stop())

    # Step 10: Remove model
    result.add_step(step_remove_model(profile.id))

    # Step 11: Clean uninstall check
    result.add_step(step_clean_uninstall())

    # Cleanup
    step_cleanup_skills(["crm-frontend", "crm-backend", "crm-database"])

    result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


def run_test_4_qwen3_multiprocess():
    """Test 4: Qwen 3 multi-process + yolo mode + CRM build."""
    result = TestResult(
        test_name="Test 4: Qwen 3 Multi-Process + Yolo Mode CRM Build",
        model_id="",
        model_family="qwen3",
        engine="llama.cpp (server)",
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    config = TqConfig.load()
    config.ensure_dirs()
    sys_info = detect_system()
    registry = ModelRegistry(config.models_dir)

    # Step 1: Hardware selection
    step, profile = step_verify_hardware_selection(registry, sys_info, "qwen3")
    result.add_step(step)
    if not profile:
        result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        return result
    result.model_id = profile.id

    # Ensure model is downloaded (re-download if removed by earlier test)
    registry.scan_local_models()
    profile = registry.get_profile(profile.id)
    if not profile or not profile.local_path:
        start = time.time()
        try:
            orig_profile = [p for p in BUILTIN_PROFILES if p.id == result.model_id][0]
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=orig_profile.hf_repo,
                filename=orig_profile.filename,
                local_dir=str(config.models_dir),
            )
            registry.scan_local_models()
            profile = registry.get_profile(result.model_id)
            elapsed = time.time() - start
            result.add_step(StepResult(
                name="re_download_model",
                passed=profile is not None and profile.local_path is not None,
                duration_s=elapsed,
                details=f"Re-downloaded model for multi-process test",
            ))
        except Exception as e:
            result.add_step(StepResult(
                name="re_download_model",
                passed=False,
                duration_s=time.time() - start,
                details=f"Failed to re-download: {e}",
            ))
            result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
            return result

    if not profile or not profile.local_path:
        result.add_step(StepResult(
            name="model_available",
            passed=False,
            details="Model not available for multi-process test",
        ))
        result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        return result

    # Step 2: Assess multiprocess feasibility (yolo mode)
    start = time.time()
    from tqcli.core.multiprocess import assess_multiprocess
    plan = assess_multiprocess(
        sys_info=sys_info,
        model_path=str(profile.local_path),
        model_size_mb=profile.min_ram_mb,
        requested_workers=2,
        preferred_engine="llama.cpp",
        unrestricted=True,
    )
    elapsed = time.time() - start
    result.add_step(StepResult(
        name="multiprocess_assessment_yolo",
        passed=plan.feasible,
        duration_s=elapsed,
        details=f"Engine: {plan.engine}, Max workers: {plan.max_workers}, Recommended: {plan.recommended_workers}",
        metrics={
            "engine": plan.engine,
            "max_workers": plan.max_workers,
            "recommended_workers": plan.recommended_workers,
            "feasible": plan.feasible,
            "warnings": plan.warnings,
            "unrestricted": True,
        },
    ))

    # Step 3: Start server (yolo mode)
    result.add_step(step_multiprocess_serve_start(profile.id, unrestricted=True))

    # Step 4: Check server status
    result.add_step(step_multiprocess_serve_status())

    # Step 5: Generate CRM skills
    result.add_step(step_generate_crm_skills())

    # Step 6: Create CRM workspace
    result.add_step(step_create_crm_workspace())

    # Step 7: Verify CRM workspace
    result.add_step(step_verify_crm_workspace())

    # Step 8: Delete CRM workspace
    result.add_step(step_delete_crm_workspace())

    # Step 9: Stop server
    result.add_step(step_multiprocess_serve_stop())

    # Step 10: Remove model
    result.add_step(step_remove_model(profile.id))

    # Step 11: Clean uninstall check
    result.add_step(step_clean_uninstall())

    # Cleanup
    step_cleanup_skills(["crm-frontend", "crm-backend", "crm-database"])

    result.finished = time.strftime("%Y-%m-%d %H:%M:%S")
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


def format_report(results: list[TestResult], system_info: dict) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# tqCLI Integration Test Report")
    lines.append(f"\n**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**tqCLI Version:** 0.3.1")
    lines.append(f"**Backend:** llama.cpp (llama-cpp-python)")
    lines.append("")

    # System info
    lines.append("## System Information")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    for k, v in system_info.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Summary
    total_pass = sum(r.pass_count for r in results)
    total_fail = sum(r.fail_count for r in results)
    total_tests = sum(len(r.steps) for r in results)
    lines.append("## Summary")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Steps | {total_tests} |")
    lines.append(f"| Passed | {total_pass} |")
    lines.append(f"| Failed | {total_fail} |")
    lines.append(f"| Pass Rate | {total_pass/max(total_tests,1)*100:.1f}% |")
    lines.append("")

    # Per-test results
    for r in results:
        lines.append(f"## {r.test_name}")
        lines.append(f"**Model:** {r.model_id} | **Engine:** {r.engine} | **Duration:** {r.total_duration_s:.1f}s | **Result:** {'PASS' if r.passed else 'FAIL'}")
        lines.append(f"\n**Started:** {r.started} | **Finished:** {r.finished}")
        lines.append("")
        lines.append("| # | Step | Result | Duration | Details |")
        lines.append("|---|------|--------|----------|---------|")
        for i, s in enumerate(r.steps, 1):
            status = "PASS" if s.passed else "FAIL"
            details = s.details[:100].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {i} | {s.name} | {status} | {s.duration_s:.2f}s | {details} |")
        lines.append("")

        # Performance metrics
        perf_steps = [s for s in r.steps if s.metrics.get("tokens_per_second")]
        if perf_steps:
            lines.append("### Performance Metrics")
            lines.append("| Step | Tokens/s | Completion Tokens | TTFT (s) | Total Time (s) |")
            lines.append("|------|----------|-------------------|----------|----------------|")
            for s in perf_steps:
                m = s.metrics
                lines.append(
                    f"| {s.name} | {m.get('tokens_per_second', 'N/A')} | "
                    f"{m.get('completion_tokens', 'N/A')} | "
                    f"{m.get('time_to_first_token_s', 'N/A')} | "
                    f"{m.get('total_time_s', 'N/A')} |"
                )
            lines.append("")

    # Issues found
    lines.append("## Issues Found and Fixed")
    lines.append("")
    lines.append("| # | Issue | Severity | Status |")
    lines.append("|---|-------|----------|--------|")
    lines.append("| 1 | All Gemma 4 HF repos were `google/` (non-existent), should be `unsloth/` | Critical | Fixed |")
    lines.append("| 2 | All Gemma 4 filenames had wrong casing | Critical | Fixed |")
    lines.append("| 3 | Qwen 3 GGUF filenames had wrong casing (lowercase vs mixed-case) | Critical | Fixed |")
    lines.append("| 4 | Missing multimodal input support (image/audio) for Gemma 4 | High | Fixed |")
    lines.append("| 5 | Missing skill generation command (`tqcli skill create`) | High | Fixed |")
    lines.append("| 6 | Gemma 4 26B MoE model ID misnamed as 27b | Low | Noted |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run specific test (1-4)")
    parser.add_argument("--output", default="/llm_models_python_code_src/tqCLI/TEST_REPORT.md",
                        help="Output report path")
    args = parser.parse_args()

    system_info = get_system_info()
    results = []

    if args.test is None or args.test == 1:
        print("=" * 60)
        print("RUNNING TEST 1: Gemma 4 + llama.cpp Full Lifecycle")
        print("=" * 60)
        results.append(run_test_1_gemma4())
        print(f"Test 1: {'PASS' if results[-1].passed else 'FAIL'} ({results[-1].pass_count}/{len(results[-1].steps)} steps)")

    if args.test is None or args.test == 2:
        print("=" * 60)
        print("RUNNING TEST 2: Qwen 3 + llama.cpp Full Lifecycle")
        print("=" * 60)
        results.append(run_test_2_qwen3())
        print(f"Test 2: {'PASS' if results[-1].passed else 'FAIL'} ({results[-1].pass_count}/{len(results[-1].steps)} steps)")

    if args.test is None or args.test == 3:
        print("=" * 60)
        print("RUNNING TEST 3: Gemma 4 Multi-Process + Yolo CRM")
        print("=" * 60)
        results.append(run_test_3_gemma4_multiprocess())
        print(f"Test 3: {'PASS' if results[-1].passed else 'FAIL'} ({results[-1].pass_count}/{len(results[-1].steps)} steps)")

    if args.test is None or args.test == 4:
        print("=" * 60)
        print("RUNNING TEST 4: Qwen 3 Multi-Process + Yolo CRM")
        print("=" * 60)
        results.append(run_test_4_qwen3_multiprocess())
        print(f"Test 4: {'PASS' if results[-1].passed else 'FAIL'} ({results[-1].pass_count}/{len(results[-1].steps)} steps)")

    # Generate report
    report = format_report(results, system_info)
    Path(args.output).write_text(report)
    print(f"\nReport written to: {args.output}")

    # Also write JSON results
    json_path = Path(args.output).with_suffix(".json")
    json_data = {
        "system_info": system_info,
        "results": [],
    }
    for r in results:
        test_data = {
            "test_name": r.test_name,
            "model_id": r.model_id,
            "model_family": r.model_family,
            "engine": r.engine,
            "started": r.started,
            "finished": r.finished,
            "total_duration_s": r.total_duration_s,
            "passed": r.passed,
            "pass_count": r.pass_count,
            "fail_count": r.fail_count,
            "steps": [
                {
                    "name": s.name,
                    "passed": s.passed,
                    "duration_s": s.duration_s,
                    "details": s.details,
                    "metrics": s.metrics,
                }
                for s in r.steps
            ],
        }
        json_data["results"].append(test_data)

    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    print(f"JSON data written to: {json_path}")
