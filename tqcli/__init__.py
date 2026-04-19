"""tqCLI — TurboQuant CLI for local LLM inference with smart routing."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tqcli")
except PackageNotFoundError:
    # Fallback if package is not installed (e.g. during development/tests)
    __version__ = "0.6.2"

__app_name__ = "tqCLI"
