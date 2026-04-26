"""tqCLI — TurboQuant CLI for local LLM inference with smart routing."""

from importlib.metadata import version, PackageNotFoundError

try:
    # PyPI distribution name is `turboquant-cli`; import name remains `tqcli`
    # (dateutil pattern). Try the published name first, fall back to a legacy
    # editable install that still uses `tqcli` as the dist name.
    try:
        __version__ = version("turboquant-cli")
    except PackageNotFoundError:
        __version__ = version("tqcli")
except PackageNotFoundError:
    # Fallback if package is not installed (e.g. during development/tests)
    __version__ = "0.7.0"

__app_name__ = "tqCLI"
