#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGPERF_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "${FLAGPERF_ROOT}/.." && pwd)"

TORCH_FL_REPO="${TORCH_FL_SOURCE_REPO:-${WORKSPACE_ROOT}/PyTorch-Plugin-FL}"
FLAG_GEMS_REPO="${FLAG_GEMS_SOURCE_REPO:-${WORKSPACE_ROOT}/FlagGems}"
TORCH_FL_COMMIT="af50297463d59ca4bb3aca59f51724afb5f6723a"
FLAG_GEMS_COMMIT="f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4"
IMAGE_TAG="flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64"
MPICH_VERSION="4.1.3"
MPICH_SHA256="4a862d87bc95c3fc5a9a49365e5cc1b6c06d72da879475817011a11e0a1c68c2"
MPICH_URL="https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz"
PYTHON_SOURCE_IMAGE="quay.io/ascend/vllm-ascend@sha256:5cf8a2b6db8b06eb1bc7fc7d191d667aebf2b197351bdba13f776918c11ec7a7"
BASE_IMAGE="harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl@sha256:a36a302204e282411dd46bd3f1edd64f7fce7190020bb4511e8b409ac8d4ba29"

require_commit() {
    local repo="$1"
    local expected="$2"
    local actual
    actual="$(git -C "${repo}" rev-parse HEAD)"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "ERROR: ${repo} is at ${actual}, expected ${expected}" >&2
        exit 2
    fi
}

require_commit "${TORCH_FL_REPO}" "${TORCH_FL_COMMIT}"
require_commit "${FLAG_GEMS_REPO}" "${FLAG_GEMS_COMMIT}"

BUILD_CONTEXT="$(mktemp -d /tmp/flagrt-ascend-image.XXXXXX)"
WHEELHOUSE_CACHE="${FLAGRT_WHEELHOUSE_CACHE:-/tmp/flagrt-ascend-wheelhouse-cache}"
SOURCE_CACHE="${FLAGRT_SOURCE_CACHE:-/tmp/flagrt-ascend-source-cache}"
cleanup() {
    chmod -R u+w "${BUILD_CONTEXT}" 2>/dev/null || true
    rm -rf "${BUILD_CONTEXT}"
}
trap cleanup EXIT

mkdir -p "${BUILD_CONTEXT}/src/Torch-FL" \
         "${BUILD_CONTEXT}/src/FlagGems" \
         "${BUILD_CONTEXT}/wheelhouse" \
         "${BUILD_CONTEXT}/third_party" \
         "${WHEELHOUSE_CACHE}" "${SOURCE_CACHE}"

MPICH_ARCHIVE="${SOURCE_CACHE}/mpich-${MPICH_VERSION}.tar.gz"
if [[ ! -f "${MPICH_ARCHIVE}" ]]; then
    curl -fL --retry 3 -o "${MPICH_ARCHIVE}.tmp" "${MPICH_URL}"
    mv "${MPICH_ARCHIVE}.tmp" "${MPICH_ARCHIVE}"
fi
echo "${MPICH_SHA256}  ${MPICH_ARCHIVE}" | sha256sum --check --status || {
    echo "ERROR: MPICH source checksum mismatch: ${MPICH_ARCHIVE}" >&2
    exit 4
}
cp "${MPICH_ARCHIVE}" "${BUILD_CONTEXT}/third_party/"

cp "${SCRIPT_DIR}/Dockerfile" \
   "${SCRIPT_DIR}/flag_gems_fused_dsa_init.py" \
   "${SCRIPT_DIR}/requirements-runtime.txt" \
   "${SCRIPT_DIR}/patch_triton_ascend_3_2_1.py" \
   "${SCRIPT_DIR}/verify_runtime.py" \
   "${BUILD_CONTEXT}/"

git -C "${TORCH_FL_REPO}" archive "${TORCH_FL_COMMIT}" \
    | tar -x -C "${BUILD_CONTEXT}/src/Torch-FL"
git -C "${FLAG_GEMS_REPO}" archive "${FLAG_GEMS_COMMIT}" \
    | tar -x -C "${BUILD_CONTEXT}/src/FlagGems"

cp -a "${WHEELHOUSE_CACHE}/." "${BUILD_CONTEXT}/wheelhouse/"

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "${BUILD_CONTEXT}:/build" \
    --entrypoint /bin/bash \
    "${PYTHON_SOURCE_IMAGE}" \
    -lc 'python3 -m pip download --dest /build/wheelhouse \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
        -r /build/requirements-runtime.txt \
        "setuptools>=64,<77" "setuptools-scm>=8,<10" wheel==0.46.2 cmake ninja'

cp -a "${BUILD_CONTEXT}/wheelhouse/." "${WHEELHOUSE_CACHE}/"

if find "${BUILD_CONTEXT}/wheelhouse" -maxdepth 1 -type f \
    -iname '*torch*npu*' | grep -q .; then
    echo "ERROR: wheelhouse unexpectedly contains torch_npu" >&2
    exit 3
fi

(cd "${BUILD_CONTEXT}" && sha256sum wheelhouse/* third_party/* | sort -k2) \
    > "${BUILD_CONTEXT}/artifacts.sha256"
cp "${BUILD_CONTEXT}/artifacts.sha256" "${SCRIPT_DIR}/artifacts.sha256"

docker build \
    --network=none \
    --pull=false \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "PYTHON_SOURCE_IMAGE=${PYTHON_SOURCE_IMAGE}" \
    --build-arg "TORCH_FL_COMMIT=${TORCH_FL_COMMIT}" \
    --build-arg "FLAG_GEMS_COMMIT=${FLAG_GEMS_COMMIT}" \
    --build-arg "MPICH_VERSION=${MPICH_VERSION}" \
    --tag "${IMAGE_TAG}" \
    "${BUILD_CONTEXT}"

IMAGE_ID="$(docker image inspect "${IMAGE_TAG}" --format '{{.Id}}')"
CONFIG_DIGEST="$(docker image inspect "${IMAGE_TAG}" --format '{{.Config.Image}}')"
cat > "${SCRIPT_DIR}/image-manifest.json" <<EOF
{
  "schema_version": 1,
  "image": "${IMAGE_TAG}",
  "image_id": "${IMAGE_ID}",
  "config_image": "${CONFIG_DIGEST}",
  "platform": "linux/arm64",
  "base": "${BASE_IMAGE}",
  "python_source": "${PYTHON_SOURCE_IMAGE}",
  "torch_fl_commit": "${TORCH_FL_COMMIT}",
  "flag_gems_commit": "${FLAG_GEMS_COMMIT}",
  "mpich_version": "${MPICH_VERSION}",
  "mpich_source_sha256": "${MPICH_SHA256}",
  "validated": false
}
EOF

echo "Built ${IMAGE_TAG} (${IMAGE_ID})"
