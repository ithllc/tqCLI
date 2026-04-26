#!/usr/bin/env bash
# scripts/_build_one_wheel.sh
#
# Builds ONE vllm-turboquant wheel on the GCP build VM. Invoked by
# build_wheel_gcp.sh once per (flavour, python-version) combination.
#
# Args:
#   $1 = flavour          (e.g. "ampere-ada-hopper" or "blackwell")
#   $2 = python version   (e.g. "3.10")
#   $3 = distribution name (e.g. "vllm-turboquant" or "vllm-turboquant-blackwell")
#   $4 = TORCH_CUDA_ARCH_LIST literal (e.g. "8.0 8.6 8.9 9.0")
#   $5 = GCS bucket URI    (e.g. "gs://tqcli-wheel-build")
#   $6 = GCS prefix        (e.g. "0.7.0-tq1")
#
# Side effects:
#   * Restores pyproject.toml + vllm/__init__.py to source state.
#   * Edits both files in place for this build.
#   * Builds the wheel into ~/vllm-turboquant/dist/.
#   * Uploads the wheel + sha256 to ${GCS_BUCKET}/${GCS_PREFIX}/.
#   * Hard-fails if the wheel exceeds 2 GiB (per Workstream B step 8).
#
set -euo pipefail

flavour="$1"
pyver="$2"
dist_name="$3"
arch_list="$4"
gcs_bucket="$5"
gcs_prefix="$6"

source /etc/profile.d/cuda.sh
cd "$HOME/vllm-turboquant"

# Restore source state so the previous build's edits do not leak in.
git checkout pyproject.toml vllm/__init__.py

# 1. Set wheel name in pyproject.toml
sed -i "s/VLLM_TURBOQUANT_NAME_PLACEHOLDER/${dist_name}/g" pyproject.toml

# 2. Inject TURBOQUANT_BUILD_ARCH + TURBOQUANT_BUILD_ARCH_LIST values
python3 - "${flavour}" "${arch_list}" <<'EOPY'
import re, sys, pathlib
flavour, arch_list = sys.argv[1], sys.argv[2]
p = pathlib.Path("vllm/__init__.py")
src = p.read_text()
src = re.sub(r'^TURBOQUANT_BUILD_ARCH\s*=.*$',
             f'TURBOQUANT_BUILD_ARCH = "{flavour}"', src, count=1, flags=re.M)
src = re.sub(r'^TURBOQUANT_BUILD_ARCH_LIST\s*=.*$',
             f'TURBOQUANT_BUILD_ARCH_LIST = "{arch_list}"', src, count=1, flags=re.M)
p.write_text(src)
print("arch fields injected")
EOPY

# 3. Set up venv for this Python version
rm -rf ".build-venv-${pyver}"
"python${pyver}" -m venv ".build-venv-${pyver}"
# shellcheck disable=SC1091
source ".build-venv-${pyver}/bin/activate"
pip install --upgrade pip build wheel setuptools "torch>=2.4" ninja

# 4. Build env
export MAX_JOBS=4
export NVCC_THREADS=4
export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST="${arch_list}"
export CMAKE_BUILD_PARALLEL_LEVEL=4
export CC="ccache gcc"
export CXX="ccache g++"

# 5. Build
rm -rf dist build
python -m build --wheel --no-isolation --outdir dist/

# 6. Verify wheel
WHEEL="$(ls dist/*.whl)"
WHEEL_SIZE_MIB="$(du -m "${WHEEL}" | cut -f1)"
echo "BUILT ${WHEEL} (${WHEEL_SIZE_MIB} MiB)"
if [[ "${WHEEL_SIZE_MIB}" -gt 2048 ]]; then
    echo "FATAL: wheel exceeds 2 GiB single-file limit (${WHEEL_SIZE_MIB} MiB)" >&2
    echo "${WHEEL}" > dist/OVERSIZED
    exit 1
fi

# 7. SHA256 + upload
( cd dist && sha256sum "$(basename "${WHEEL}")" > "$(basename "${WHEEL}").sha256" )
gsutil cp dist/*.whl     "${gcs_bucket}/${gcs_prefix}/"
gsutil cp dist/*.sha256  "${gcs_bucket}/${gcs_prefix}/"
echo "UPLOADED ${flavour}-${pyver}"
deactivate
