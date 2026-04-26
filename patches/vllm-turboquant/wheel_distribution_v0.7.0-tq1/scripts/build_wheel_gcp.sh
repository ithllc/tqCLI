#!/usr/bin/env bash
# scripts/build_wheel_gcp.sh
#
# Build the six vllm-turboquant wheels for v0.7.0-tq1 sequentially on a single
# GCP n2-standard-8 VM, push each to gs://tqcli-wheel-build/0.7.0-tq1/, and tear
# the VM down on success.
#
# Wheel matrix (locked 2026-04-26):
#   flavour=ampere-ada-hopper   TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
#   flavour=blackwell           TORCH_CUDA_ARCH_LIST="10.0 12.0 12.1+PTX"
#   python: 3.10, 3.11, 3.12
#
# Wall time ~30h, compute ~$11.64. Stays inside the default 8-vCPU regional
# quota; no quota request needed.
#
# Usage (run on the maintainer's local box; the script SSHes to GCP):
#
#     bash scripts/build_wheel_gcp.sh                     # full run, tear down on success
#     bash scripts/build_wheel_gcp.sh --keep-vm-on-error  # leave VM up if any build fails
#     bash scripts/build_wheel_gcp.sh --resume-from blackwell-3.11
#
# Prerequisites on the local box:
#   * gcloud CLI authenticated against tqcli-wheel-build
#   * gsutil available
#   * git remote `origin` -> github.com/tqcli/vllm-turboquant
#
set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
PROJECT="${TQ_GCP_PROJECT:-tqcli-wheel-build}"
ZONE="${TQ_GCP_ZONE:-us-central1-a}"
VM_NAME="${TQ_VM_NAME:-vllm-tq-builder}"
MACHINE="${TQ_MACHINE:-n2-standard-8}"
DISK_SIZE_GB="${TQ_DISK_SIZE_GB:-200}"
IMAGE_FAMILY="${TQ_IMAGE_FAMILY:-ubuntu-2204-lts}"
IMAGE_PROJECT="${TQ_IMAGE_PROJECT:-ubuntu-os-cloud}"
RELEASE_TAG="${TQ_RELEASE_TAG:-v0.7.0-tq1}"
GCS_BUCKET="${TQ_GCS_BUCKET:-gs://tqcli-wheel-build}"
GCS_PREFIX="${TQ_GCS_PREFIX:-0.7.0-tq1}"
FORK_REPO="${TQ_FORK_REPO:-https://github.com/tqcli/vllm-turboquant.git}"

KEEP_VM_ON_ERROR=0
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-vm-on-error) KEEP_VM_ON_ERROR=1; shift ;;
        --resume-from)      RESUME_FROM="$2"; shift 2 ;;
        --help|-h)
            sed -n "1,40p" "$0"
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER_SCRIPT="${SCRIPT_DIR}/_build_one_wheel.sh"
[[ -f "${HELPER_SCRIPT}" ]] || { echo "missing helper: ${HELPER_SCRIPT}" >&2; exit 2; }

# (flavour, py-version) sequence. Cheaper Ampere wheels first so a failure on
# the Blackwell side does not waste compute earlier.
BUILD_MATRIX=(
    "ampere-ada-hopper:3.10"
    "ampere-ada-hopper:3.11"
    "ampere-ada-hopper:3.12"
    "blackwell:3.10"
    "blackwell:3.11"
    "blackwell:3.12"
)

flavour_dist() {
    case "$1" in
        ampere-ada-hopper) echo "vllm-turboquant" ;;
        blackwell)         echo "vllm-turboquant-blackwell" ;;
        *) echo "" ;;
    esac
}
flavour_archs() {
    case "$1" in
        ampere-ada-hopper) echo "8.0 8.6 8.9 9.0" ;;
        blackwell)         echo "10.0 12.0 12.1+PTX" ;;
        *) echo "" ;;
    esac
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log() { printf "[build_wheel_gcp] %s\n" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }

teardown_vm() {
    log "tearing down VM ${VM_NAME}"
    gcloud compute instances delete "${VM_NAME}" \
        --project="${PROJECT}" --zone="${ZONE}" --quiet || true
}

trap_handler() {
    rc=$?
    if [[ $rc -ne 0 && $KEEP_VM_ON_ERROR -eq 1 ]]; then
        log "build failed (rc=$rc); leaving VM up per --keep-vm-on-error"
        log "ssh: gcloud compute ssh ${VM_NAME} --project=${PROJECT} --zone=${ZONE}"
        exit $rc
    fi
    teardown_vm
    exit $rc
}
trap trap_handler EXIT

remote() {
    gcloud compute ssh "${VM_NAME}" \
        --project="${PROJECT}" --zone="${ZONE}" \
        --quiet --command="$1"
}

# ----------------------------------------------------------------------------
# 1. Provision VM if missing
# ----------------------------------------------------------------------------
if gcloud compute instances describe "${VM_NAME}" \
        --project="${PROJECT}" --zone="${ZONE}" --quiet >/dev/null 2>&1; then
    log "VM ${VM_NAME} already exists; reusing"
else
    log "creating VM ${VM_NAME} (${MACHINE}, ${DISK_SIZE_GB}GB ${IMAGE_FAMILY})"
    gcloud compute instances create "${VM_NAME}" \
        --project="${PROJECT}" --zone="${ZONE}" \
        --machine-type="${MACHINE}" \
        --image-family="${IMAGE_FAMILY}" --image-project="${IMAGE_PROJECT}" \
        --boot-disk-size="${DISK_SIZE_GB}GB" --boot-disk-type="pd-balanced" \
        --scopes="https://www.googleapis.com/auth/devstorage.read_write,https://www.googleapis.com/auth/compute"
    log "waiting up to 60s for SSH"
    for _ in $(seq 1 12); do
        if gcloud compute ssh "${VM_NAME}" --project="${PROJECT}" --zone="${ZONE}" \
                --quiet --command="true" 2>/dev/null; then
            break
        fi
        sleep 5
    done
fi

# ----------------------------------------------------------------------------
# 2. Bootstrap toolchain on VM (idempotent)
# ----------------------------------------------------------------------------
log "running bootstrap on VM (idempotent; no-op if already done)"
remote 'set -e; if [[ -f $HOME/.tq-bootstrap-done ]]; then echo bootstrap-already-done; exit 0; fi
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential ccache git curl wget ninja-build \
    software-properties-common ca-certificates gnupg
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    python3.11 python3.11-venv python3.11-dev \
    python3.12 python3.12-venv python3.12-dev
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-13-0
sudo tee /etc/profile.d/cuda.sh >/dev/null <<EOPROFILE
export PATH=/usr/local/cuda-13.0/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:\${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-13.0
EOPROFILE
ccache -M 50G
touch $HOME/.tq-bootstrap-done
echo bootstrap-complete'

# ----------------------------------------------------------------------------
# 3. Sync fork checkout at the release tag
# ----------------------------------------------------------------------------
log "syncing fork checkout to ${RELEASE_TAG}"
remote "set -e
if [[ ! -d \$HOME/vllm-turboquant ]]; then git clone ${FORK_REPO} \$HOME/vllm-turboquant; fi
cd \$HOME/vllm-turboquant
git fetch --tags --force origin
git checkout ${RELEASE_TAG}
echo HEAD: \$(git rev-parse --short HEAD)"

# ----------------------------------------------------------------------------
# 4. Upload helper script
# ----------------------------------------------------------------------------
log "uploading helper script to VM"
gcloud compute scp "${HELPER_SCRIPT}" \
    "${VM_NAME}":~/_build_one_wheel.sh \
    --project="${PROJECT}" --zone="${ZONE}" --quiet
remote "chmod +x ~/_build_one_wheel.sh"

# ----------------------------------------------------------------------------
# 5. Build loop
# ----------------------------------------------------------------------------
SHOULD_RUN=1
if [[ -n "${RESUME_FROM}" ]]; then SHOULD_RUN=0; fi

for entry in "${BUILD_MATRIX[@]}"; do
    flavour="${entry%%:*}"
    pyver="${entry##*:}"
    label="${flavour}-${pyver}"

    if [[ "${SHOULD_RUN}" -eq 0 ]]; then
        if [[ "${label}" == "${RESUME_FROM}" ]]; then
            SHOULD_RUN=1
        else
            log "skipping ${label} (resume-from=${RESUME_FROM})"
            continue
        fi
    fi

    dist_name="$(flavour_dist "${flavour}")"
    arch_list="$(flavour_archs "${flavour}")"
    [[ -n "${dist_name}" ]] || die "unknown flavour ${flavour}"

    log "BUILD ${label}: dist=${dist_name} archs='${arch_list}'"
    remote "bash ~/_build_one_wheel.sh '${flavour}' '${pyver}' '${dist_name}' '${arch_list}' '${GCS_BUCKET}' '${GCS_PREFIX}'"
    log "completed ${label}"
done

# ----------------------------------------------------------------------------
# 6. Summary
# ----------------------------------------------------------------------------
log "all builds complete; listing GCS prefix"
gsutil ls "${GCS_BUCKET}/${GCS_PREFIX}/" | tee /tmp/tq_wheel_manifest.txt

count=$(grep -c '\.whl$' /tmp/tq_wheel_manifest.txt || echo 0)
if [[ "${count}" -ne 6 ]]; then
    die "expected 6 wheels in GCS, found ${count}"
fi

log "SUCCESS: 6 wheels at ${GCS_BUCKET}/${GCS_PREFIX}/"
log "next: 'gsutil cp ${GCS_BUCKET}/${GCS_PREFIX}/* ./dist/' then 'gh release create ${RELEASE_TAG}' (see docs/RELEASING.md)"
