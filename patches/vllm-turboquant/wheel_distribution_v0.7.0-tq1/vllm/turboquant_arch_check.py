"""Runtime arch-vs-build compatibility check for TurboQuant vLLM wheels.

The 0.7.0 release ships two flavoured wheels:

    vllm-turboquant            -> sm 8.0 / 8.6 / 8.9 / 9.0   (Ampere/Ada/Hopper)
    vllm-turboquant-blackwell  -> sm 10.0 / 12.0 / 12.1+PTX (Blackwell DC, consumer, GB10)

A user can install the wrong flavour for their GPU. This module hard-fails on
the first GPU init with a clear remediation message instead of silently falling
back, garbling kernels, or producing nonsense outputs.

Wired in by ``vllm`` engine startup: callers should invoke
``assert_arch_compatibility()`` once per process before launching the worker.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


_FLAVOR_ALT = {
    "ampere-ada-hopper": "vllm-turboquant-blackwell",
    "blackwell": "vllm-turboquant",
}


def _parse_arch_list(arch_list: str) -> List[Tuple[int, int]]:
    """Parse a TORCH_CUDA_ARCH_LIST string into (major, minor) tuples.

    Handles ``"8.0 8.6 8.9 9.0"`` and ``"10.0 12.0 12.1+PTX"``. The +PTX suffix
    expresses forward-compat (PTX JIT) but does not change the compiled SASS
    target, so we treat it as the same compute capability.
    """
    archs: List[Tuple[int, int]] = []
    for token in arch_list.split():
        token = token.strip().split("+", 1)[0]
        if not token:
            continue
        try:
            major_str, minor_str = token.split(".")
            archs.append((int(major_str), int(minor_str)))
        except (ValueError, IndexError):
            # Skip malformed tokens; build script controls this string.
            continue
    return archs


def _detected_capability() -> Optional[Tuple[Tuple[int, int], str]]:
    """Return ((major, minor), device_name) for cuda:0, or None if unavailable."""
    try:
        import torch  # local import: keep this module importable on CPU-only boxes
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    return (int(cap[0]), int(cap[1])), name


def check_arch_compatibility() -> Optional[str]:
    """Return a remediation message if the wheel/runtime arches mismatch, else None.

    Returns ``None`` when:
      * No CUDA GPU is visible (CPU-only smoke tests, container without --gpus).
      * The build sentinels are empty (editable source install — no wheel claim).
      * The detected GPU's compute capability is in the build's arch list.
    """
    cap_info = _detected_capability()
    if cap_info is None:
        return None
    capability, device_name = cap_info

    try:
        from vllm import TURBOQUANT_BUILD_ARCH, TURBOQUANT_BUILD_ARCH_LIST
    except ImportError:
        return None

    if not TURBOQUANT_BUILD_ARCH or not TURBOQUANT_BUILD_ARCH_LIST:
        # Editable / source build with no wheel-time arch claim. Skip.
        return None

    built_archs = _parse_arch_list(TURBOQUANT_BUILD_ARCH_LIST)
    if not built_archs:
        return None
    if capability in built_archs:
        return None

    built_str = " / ".join(f"sm_{m}.{n}" for m, n in built_archs)
    detected_str = f"sm_{capability[0]}.{capability[1]}"
    recommend = _FLAVOR_ALT.get(TURBOQUANT_BUILD_ARCH, "the matching vllm-turboquant flavour")

    return (
        f"This wheel was built for {TURBOQUANT_BUILD_ARCH} GPUs ({built_str}). "
        f"Detected your GPU as {detected_str} ({device_name}). "
        f"Install `{recommend}` instead. "
        f"See https://github.com/tqcli/vllm-turboquant#install for the GPU-flavour table."
    )


def assert_arch_compatibility() -> None:
    """Raise ``RuntimeError`` if the wheel/runtime arches mismatch.

    Idempotent and cheap — designed to be called from engine startup. No-op when
    no GPU is visible or when running from an editable source tree.
    """
    msg = check_arch_compatibility()
    if msg is not None:
        raise RuntimeError(msg)
