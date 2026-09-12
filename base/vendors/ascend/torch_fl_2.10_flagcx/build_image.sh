#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGPERF_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "${FLAGPERF_ROOT}/.." && pwd)"

FLAGCX_REPO="${FLAGCX_SOURCE_REPO:-${WORKSPACE_ROOT}/FlagCX}"
FLAGCX_JSON_REPO="${FLAGCX_JSON_SOURCE_REPO:-${FLAGCX_REPO}/third-party/json}"
FLAGCX_COMMIT="55eb2ffff6988ae1db5e6ecb325472aecc93d238"
FLAGCX_JSON_COMMIT="a0e9fb1e638cfbb5b8b556b7c51eaa81977bad48"
FLAGCX_VERSION="0.13.0"
BASE_IMAGE="flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64"
BASE_IMAGE_ID="sha256:d948410966b0dfdfaf4f9c95b9b14bdb7c4279cb85ccd7f9157f0fb5d1c6d397"
IMAGE_TAG="flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64"
FLAGCX_PATCH="${SCRIPT_DIR}/patches/0001-ascend-use-exported-flagos-stream-api.patch"
FLAGCX_PATCH_SHA256="bc84ea26eee3783085ac140e8792455433299fb0bb2330ec9d2680b6d2b7736d"
FLAGCX_EVENT_PATCH="${SCRIPT_DIR}/patches/0002-ascend-use-sync-event-for-cross-stream-wait.patch"
FLAGCX_EVENT_PATCH_SHA256="524654689fdfe896c2f935f89e884bca4c57e50cd263cc145659e3b8b8ac7f0a"
VERIFIER_SHA256="$(sha256sum "${SCRIPT_DIR}/verify_flagcx_p2p.py" | cut -d ' ' -f 1)"

require_git_commit() {
    local repository="$1"
    local commit="$2"
    local label="$3"
    if ! git -C "${repository}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        echo "ERROR: ${label} commit ${commit} is unavailable in ${repository}" >&2
        exit 2
    fi
}

require_git_commit "${FLAGCX_REPO}" "${FLAGCX_COMMIT}" "FlagCX"
if [[ ! -e "${FLAGCX_JSON_REPO}/.git" ]]; then
    echo "ERROR: initialize the pinned FlagCX third-party/json submodule or set FLAGCX_JSON_SOURCE_REPO" >&2
    exit 2
fi
require_git_commit "${FLAGCX_JSON_REPO}" "${FLAGCX_JSON_COMMIT}" "nlohmann/json"
echo "${FLAGCX_PATCH_SHA256}  ${FLAGCX_PATCH}" \
    | sha256sum --check --status || {
        echo "ERROR: FlagCX downstream patch checksum drifted" >&2
        exit 2
    }
echo "${FLAGCX_EVENT_PATCH_SHA256}  ${FLAGCX_EVENT_PATCH}" \
    | sha256sum --check --status || {
        echo "ERROR: FlagCX event patch checksum drifted" >&2
        exit 2
    }

ACTUAL_BASE_ID="$(docker image inspect "${BASE_IMAGE}" --format '{{.Id}}')"
if [[ "${ACTUAL_BASE_ID}" != "${BASE_IMAGE_ID}" ]]; then
    echo "ERROR: parent runtime drift: actual=${ACTUAL_BASE_ID}, locked=${BASE_IMAGE_ID}" >&2
    exit 3
fi
if ! docker run --rm --network=none --entrypoint /bin/true "${BASE_IMAGE}"; then
    echo "ERROR: Docker cannot create an offline container from the locked parent; fix Docker storage/runtime health before building" >&2
    exit 4
fi

BUILD_CONTEXT="$(mktemp -d /tmp/flagrt-ascend-flagcx.XXXXXX)"
cleanup() {
    chmod -R u+w "${BUILD_CONTEXT}" 2>/dev/null || true
    rm -rf "${BUILD_CONTEXT}"
}
trap cleanup EXIT

mkdir -p "${BUILD_CONTEXT}/src/FlagCX/third-party/json"
git -C "${FLAGCX_REPO}" archive "${FLAGCX_COMMIT}" \
    | tar -x -C "${BUILD_CONTEXT}/src/FlagCX"
git -C "${FLAGCX_JSON_REPO}" archive "${FLAGCX_JSON_COMMIT}" \
    | tar -x -C "${BUILD_CONTEXT}/src/FlagCX/third-party/json"
git -C "${BUILD_CONTEXT}/src/FlagCX" apply --check "${FLAGCX_PATCH}"
git -C "${BUILD_CONTEXT}/src/FlagCX" apply "${FLAGCX_PATCH}"
git -C "${BUILD_CONTEXT}/src/FlagCX" apply --check "${FLAGCX_EVENT_PATCH}"
git -C "${BUILD_CONTEXT}/src/FlagCX" apply "${FLAGCX_EVENT_PATCH}"

cp "${SCRIPT_DIR}/Dockerfile" \
   "${SCRIPT_DIR}/verify_flagcx_runtime.py" \
   "${SCRIPT_DIR}/verify_flagcx_p2p.py" \
   "${BUILD_CONTEXT}/"

(
    cd "${BUILD_CONTEXT}/src/FlagCX"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${BUILD_CONTEXT}/source-artifacts.sha256"
cp "${BUILD_CONTEXT}/source-artifacts.sha256" \
   "${SCRIPT_DIR}/source-artifacts.sha256"

docker build \
    --network=none \
    --pull=false \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "PARENT_IMAGE_ID=${BASE_IMAGE_ID}" \
    --build-arg "FLAGCX_COMMIT=${FLAGCX_COMMIT}" \
    --build-arg "FLAGCX_VERSION=${FLAGCX_VERSION}" \
    --build-arg "FLAGCX_PATCH_SHA256=${FLAGCX_PATCH_SHA256}" \
    --build-arg "FLAGCX_EVENT_PATCH_SHA256=${FLAGCX_EVENT_PATCH_SHA256}" \
    --build-arg "VERIFIER_SHA256=${VERIFIER_SHA256}" \
    --tag "${IMAGE_TAG}" \
    "${BUILD_CONTEXT}"

IMAGE_ID="$(docker image inspect "${IMAGE_TAG}" --format '{{.Id}}')"
WHEEL_SHA256="$(docker run --rm --network=none --entrypoint /bin/bash "${IMAGE_TAG}" -lc 'cut -d " " -f 1 /opt/flagrt/flagcx-wheel.sha256')"
SOURCE_MANIFEST_SHA256="$(sha256sum "${SCRIPT_DIR}/source-artifacts.sha256" | cut -d ' ' -f 1)"

python3 - "${SCRIPT_DIR}/image-manifest.json" \
    "${IMAGE_TAG}" "${IMAGE_ID}" "${BASE_IMAGE}" "${BASE_IMAGE_ID}" \
    "${FLAGCX_COMMIT}" "${FLAGCX_JSON_COMMIT}" "${WHEEL_SHA256}" \
    "${SOURCE_MANIFEST_SHA256}" "${FLAGCX_PATCH_SHA256}" \
    "${FLAGCX_EVENT_PATCH_SHA256}" "${VERIFIER_SHA256}" <<'PY'
import json
from pathlib import Path
import sys

(
    output, image, image_id, parent_image, parent_image_id,
    flagcx_commit, json_commit, wheel_sha256, source_manifest_sha256,
    patch_sha256, event_patch_sha256, verifier_sha256,
) = sys.argv[1:]
manifest = {
    "schema_version": 1,
    "release_stage": "candidate",
    "image": image,
    "image_id": image_id,
    "platform": "linux/arm64",
    "parent_image": parent_image,
    "parent_image_id": parent_image_id,
    "flagcx_version": "0.13.0",
    "flagcx_commit": flagcx_commit,
    "flagcx_json_commit": json_commit,
    "flagcx_adaptor": "ascend",
    "flagcx_torch_backend": "flagos",
    "flagcx_downstream_patches": [{
        "path": "patches/0001-ascend-use-exported-flagos-stream-api.patch",
        "sha256": patch_sha256,
    }, {
        "path": "patches/0002-ascend-use-sync-event-for-cross-stream-wait.patch",
        "sha256": event_patch_sha256,
    }],
    "wheel_sha256": wheel_sha256,
    "source_artifacts_sha256": source_manifest_sha256,
    "qualification_verifier_sha256": verifier_sha256,
    "validated": False,
    "validation_status": "static-passed",
    "validation_scope": "Image build and static ABI/linkage checks only; no NPU was mapped.",
}
Path(output).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "Built candidate ${IMAGE_TAG} (${IMAGE_ID}); physical-NPU gates remain pending."
