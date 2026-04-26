#!/usr/bin/env bash
# community_verify.sh — friend-of-the-project Mac verification helper for
# tqCLI 0.7.0 (TurboQuant fork wheels). Prints an explicit consent manifest
# BEFORE collecting any data, exits non-zero if consent is declined.
#
# Modes:
#   --auto-report   File a GitHub issue using the user's `gh` CLI (no tokens
#                   are ever shipped to or read by this script).
#   --manual        Print the markdown block for the user to paste into the
#                   pre-filled issue at:
#                     https://github.com/tqcli/tqcli/issues/new?template=community_verify_0_7_0.yml
#   --consent-only  Print the manifest, prompt for [y/N], then exit. Lets a
#                   user audit what would be collected before running the
#                   real verifier.
#
# Idempotent: safe to re-run; staging dir is recreated each invocation.

set -euo pipefail

MODE="${1:---manual}"
SHIP_VERSION="0.7.0"
STAGE_DIR="${TMPDIR:-/tmp}/tqcli-community-verify-${SHIP_VERSION}.$$"
trap 'rm -rf "$STAGE_DIR"' EXIT

# ---------------------------------------------------------------- consent
print_manifest() {
    cat <<'MANIFEST'
=========================================================================
  tqCLI 0.7.0 — Community Verification Consent Manifest
=========================================================================

This script verifies a `pip install turboquant-cli[llama-tq]` install on
your Mac and reports the result back to the tqCLI maintainers. It is
opt-in. Nothing leaves your machine without your explicit "y" below.

WHAT IS COLLECTED:
  * macOS version       (e.g. "macOS 14.4 (Sonoma)")
  * Chip / arch         (e.g. "Apple M2", "arm64", or "Intel x86_64")
  * Python version      (e.g. "3.12.2")
  * pip install result  (success / failure, redacted exception text only)
  * `tqcli system info` output, with $HOME path replaced by "$HOME"
  * `tqcli chat --kv-quant turbo4 --prompt "Two plus two?" --json` result
    (the JSON response — it asks the model "Two plus two?", nothing else)

WHAT IS NEVER COLLECTED:
  * Username / hostname / IP address / MAC address
  * Free-form user prompts other than the canned "Two plus two?"
  * Any environment variable starting with TOKEN, KEY, SECRET, PASSWORD
  * Any file outside the tqCLI install
  * Shell history, ssh keys, or git config
  * Authentication tokens used by the `gh` CLI in --auto-report mode

MODE: %MODE%

If you proceed (`y`), the staging directory will be created at:
  %STAGE%

You will see the exact markdown block before it is filed (--auto-report)
or copied to your clipboard (--manual). You can edit or abandon at that
point.

Type `y` to consent, anything else to abort.
=========================================================================
MANIFEST
}

confirm_consent() {
    local input
    print_manifest | sed -e "s|%MODE%|${MODE}|g" -e "s|%STAGE%|${STAGE_DIR}|g"
    printf "Consent? [y/N]: "
    read -r input || input=""
    case "${input}" in
        y|Y|yes|YES) return 0 ;;
        *) printf "\n[abort] Consent declined. No data collected.\n" ; return 1 ;;
    esac
}

if [[ "${MODE}" == "--consent-only" ]]; then
    confirm_consent || exit 2
    printf "\n[ok] Consent recorded. Re-run without --consent-only to actually run.\n"
    exit 0
fi

confirm_consent || exit 2

mkdir -p "${STAGE_DIR}"

# ---------------------------------------------------------------- collect
HOME_REDACT="${HOME}"
redact() { sed -e "s|${HOME_REDACT}|\$HOME|g"; }

OS_VERSION="$(sw_vers -productVersion 2>/dev/null || uname -sr)"
OS_NAME="$(sw_vers -productName 2>/dev/null || uname -s)"
CHIP="$(uname -m)"
PYVER="$(python3 --version 2>&1 || echo 'python3 missing')"

# Use a sandboxed venv so this never trampoles a user's existing env.
VENV_DIR="${STAGE_DIR}/venv"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

PIP_LOG="${STAGE_DIR}/pip-install.log"
PIP_STATUS="ok"
if ! pip install --upgrade "turboquant-cli[llama-tq]" >"${PIP_LOG}" 2>&1; then
    PIP_STATUS="failed"
fi

SYSINFO_LOG="${STAGE_DIR}/system-info.txt"
if [[ "${PIP_STATUS}" == "ok" ]]; then
    if ! tqcli system info 2>&1 | redact >"${SYSINFO_LOG}"; then
        printf "(tqcli system info failed)\n" >"${SYSINFO_LOG}"
    fi
else
    printf "(skipped: pip install failed)\n" >"${SYSINFO_LOG}"
fi

CHAT_LOG="${STAGE_DIR}/chat-canned.json"
CHAT_STATUS="skipped"
if [[ "${PIP_STATUS}" == "ok" ]]; then
    if tqcli chat --kv-quant turbo4 --prompt "Two plus two?" --json 2>/dev/null \
        | redact >"${CHAT_LOG}"; then
        CHAT_STATUS="ok"
    else
        CHAT_STATUS="failed"
    fi
fi

# ---------------------------------------------------------------- compose
REPORT="${STAGE_DIR}/report.md"
{
    printf "## tqCLI %s — Community Verification\n\n" "${SHIP_VERSION}"
    printf -- "- **OS:** %s %s\n" "${OS_NAME}" "${OS_VERSION}"
    printf -- "- **Chip / arch:** %s\n" "${CHIP}"
    printf -- "- **Python:** %s\n" "${PYVER}"
    printf -- "- **pip install:** %s\n" "${PIP_STATUS}"
    printf -- "- **chat --json:** %s\n\n" "${CHAT_STATUS}"
    printf "### system info\n\`\`\`\n"
    cat "${SYSINFO_LOG}"
    printf "\n\`\`\`\n\n"
    printf "### chat canned response\n\`\`\`json\n"
    if [[ -s "${CHAT_LOG}" ]]; then cat "${CHAT_LOG}"; else printf "(none)"; fi
    printf "\n\`\`\`\n\n"
    if [[ "${PIP_STATUS}" != "ok" ]]; then
        printf "### pip install log (tail)\n\`\`\`\n"
        tail -50 "${PIP_LOG}" | redact
        printf "\n\`\`\`\n"
    fi
} >"${REPORT}"

printf "\n---\n[review] Report draft at: %s\n---\n\n" "${REPORT}"

# ---------------------------------------------------------------- ship
case "${MODE}" in
    --auto-report)
        if ! command -v gh >/dev/null 2>&1; then
            printf "[error] --auto-report requires the 'gh' CLI. Install from https://cli.github.com or run with --manual.\n"
            exit 3
        fi
        # NEVER pipe gh tokens to anything; gh handles its own auth state.
        gh issue create \
            --repo tqcli/tqcli \
            --title "[community-verify-${SHIP_VERSION}] ${OS_NAME} ${OS_VERSION} ${CHIP}" \
            --label "community-verify-${SHIP_VERSION}" \
            --body-file "${REPORT}"
        printf "\n[ok] Report filed via gh CLI.\n"
        ;;
    --manual)
        cat <<EOM
Paste this markdown into the pre-filled issue:

  https://github.com/tqcli/tqcli/issues/new?template=community_verify_0_7_0.yml

------------------------- BEGIN REPORT (paste below) -------------------------
EOM
        cat "${REPORT}"
        printf '\n------------------------- END REPORT -------------------------\n'
        ;;
    *)
        printf "[error] Unknown mode: %s. Use --auto-report or --manual.\n" "${MODE}"
        exit 4
        ;;
esac
