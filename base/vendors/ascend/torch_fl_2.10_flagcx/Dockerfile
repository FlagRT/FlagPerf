# syntax=docker/dockerfile:1

ARG BASE_IMAGE=flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64

FROM ${BASE_IMAGE} AS flagcx-builder

ARG FLAGCX_COMMIT
ARG FLAGCX_VERSION=0.13.0

ENV FLAGCX_ADAPTOR=ascend \
    USE_ASCEND=1 \
    FLAGCX_TORCH_BACKEND=flagos \
    ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    DEVICE_HOME=/usr/local/Ascend/ascend-toolkit/latest \
    CCL_HOME=/usr/local/Ascend/ascend-toolkit/latest \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY src/FlagCX/ /opt/flagrt/src/FlagCX/

RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && FLAGOS_ROOT="$(python3 -c 'import pathlib, torch_fl; print(pathlib.Path(torch_fl.__file__).parent)')" \
    && test -f "${FLAGOS_ROOT}/include/flagos.h" \
    && test -f "${FLAGOS_ROOT}/lib/libflagos.so" \
    && FLAGOS_INCLUDE_DIR="${FLAGOS_ROOT}/include" \
       FLAGOS_LIBRARY_DIR="${FLAGOS_ROOT}/lib" \
       python3 -m pip wheel --no-index --no-build-isolation --no-deps \
       --wheel-dir /opt/flagrt/wheels /opt/flagrt/src/FlagCX \
    && sha256sum /opt/flagrt/wheels/flagcx-${FLAGCX_VERSION}-*.whl \
       > /opt/flagrt/flagcx-wheel.sha256

FROM ${BASE_IMAGE}

ARG FLAGCX_COMMIT
ARG FLAGCX_VERSION=0.13.0
ARG PARENT_IMAGE_ID
ARG FLAGCX_PATCH_SHA256
ARG FLAGCX_EVENT_PATCH_SHA256
ARG VERIFIER_SHA256

LABEL org.opencontainers.image.title="FlagRT Ascend communication runtime" \
      org.opencontainers.image.version="0.1.3" \
      org.opencontainers.image.base.name="flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64" \
      io.flagrt.parent.image.id="${PARENT_IMAGE_ID}" \
      io.flagrt.cann.version="9.0.0" \
      io.flagrt.torch.version="2.10.0+cpu" \
      io.flagrt.flagcx.version="${FLAGCX_VERSION}" \
      io.flagrt.flagcx.commit="${FLAGCX_COMMIT}" \
      io.flagrt.flagcx.patch.sha256="${FLAGCX_PATCH_SHA256}" \
      io.flagrt.flagcx.event.patch.sha256="${FLAGCX_EVENT_PATCH_SHA256}" \
      io.flagrt.qualification.verifier.sha256="${VERIFIER_SHA256}" \
      io.flagrt.flagcx.adaptor="ascend" \
      io.flagrt.flagcx.torch_backend="flagos" \
      io.flagrt.release.stage="candidate"

ENV FLAGCX_TORCH_BACKEND=flagos \
    HCCL_WHITELIST_DISABLE=1 \
    LD_LIBRARY_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH} \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=flagcx-builder /opt/flagrt/wheels/ /opt/flagrt/wheels/
COPY --from=flagcx-builder /opt/flagrt/flagcx-wheel.sha256 /opt/flagrt/flagcx-wheel.sha256
COPY source-artifacts.sha256 /opt/flagrt/flagcx-source-artifacts.sha256
COPY verify_flagcx_runtime.py /opt/flagrt/verify_flagcx_runtime.py
COPY verify_flagcx_p2p.py /opt/flagrt/verify_flagcx_p2p.py

RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && python3 -m pip install --no-index --no-deps \
       /opt/flagrt/wheels/flagcx-${FLAGCX_VERSION}-*.whl \
    && ! python3 -m pip list --format=freeze | grep -E '^torch[-_]npu==' \
    && python3 /opt/flagrt/verify_flagcx_runtime.py --static \
    && rm -rf /opt/flagrt/wheels /root/.cache/pip

WORKDIR /workspace/FlagPerf/base
