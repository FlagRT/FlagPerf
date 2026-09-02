# syntax=docker/dockerfile:1

ARG PYTHON_SOURCE_IMAGE=quay.io/ascend/vllm-ascend@sha256:5cf8a2b6db8b06eb1bc7fc7d191d667aebf2b197351bdba13f776918c11ec7a7
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl@sha256:a36a302204e282411dd46bd3f1edd64f7fce7190020bb4511e8b409ac8d4ba29

FROM ${PYTHON_SOURCE_IMAGE} AS python-source

RUN cp -a /usr/local/python3.11.15 /opt/python3.11.15-clean \
    && rm -rf /opt/python3.11.15-clean/lib/python3.11/site-packages/*

FROM ${BASE_IMAGE}

ARG TORCH_FL_COMMIT
ARG FLAG_GEMS_COMMIT
ARG MPICH_VERSION=4.1.3

LABEL org.opencontainers.image.title="FlagRT Ascend operator runtime" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.base.digest="sha256:a36a302204e282411dd46bd3f1edd64f7fce7190020bb4511e8b409ac8d4ba29" \
      io.flagrt.cann.version="9.0.0" \
      io.flagrt.python.version="3.11.15" \
      io.flagrt.torch.version="2.10.0+cpu" \
      io.flagrt.triton_ascend.version="3.2.1" \
      io.flagrt.mpi.implementation="MPICH" \
      io.flagrt.mpi.version="${MPICH_VERSION}" \
      io.flagrt.hccl_test.collective="all_reduce" \
      io.flagrt.torch_fl.commit="${TORCH_FL_COMMIT}" \
      io.flagrt.flag_gems.commit="${FLAG_GEMS_COMMIT}"

COPY --from=python-source /opt/python3.11.15-clean /usr/local/python3.11.15

ENV PATH=/usr/local/python3.11.15/bin:/usr/local/mpich/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/python3.11.15/lib:/usr/local/mpich/lib:${LD_LIBRARY_PATH} \
    GEMS_VENDOR=ascend \
    TRITON_ENABLE_TASKQUEUE=false \
    DO_NOT_TRACK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY wheelhouse/ /opt/flagrt/wheelhouse/
COPY requirements-runtime.txt /opt/flagrt/requirements-runtime.txt
COPY third_party/mpich-${MPICH_VERSION}.tar.gz /opt/flagrt/src/mpich-${MPICH_VERSION}.tar.gz

RUN mkdir -p /opt/flagrt/src/mpich \
    && tar -xzf /opt/flagrt/src/mpich-${MPICH_VERSION}.tar.gz \
        -C /opt/flagrt/src/mpich --strip-components=1 \
    && cd /opt/flagrt/src/mpich \
    && ./configure --prefix=/usr/local/mpich --disable-fortran \
    && make -j8 \
    && make install \
    && rm -rf /opt/flagrt/src/mpich /opt/flagrt/src/mpich-${MPICH_VERSION}.tar.gz

RUN python3 -m ensurepip --upgrade \
    && python3 -m pip install --no-index --find-links=/opt/flagrt/wheelhouse \
        -r /opt/flagrt/requirements-runtime.txt \
    && ! python3 -m pip list --format=freeze | grep -E '^torch[-_]npu=='

COPY src/FlagGems/ /opt/flagrt/src/FlagGems/
COPY src/Torch-FL/ /opt/flagrt/src/Torch-FL/
COPY flag_gems_fused_dsa_init.py \
    /opt/flagrt/src/FlagGems/src/flag_gems/fused/DSA/__init__.py

RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=0.0.0+${FLAG_GEMS_COMMIT:0:12} \
       python3 -m pip install --no-index --find-links=/opt/flagrt/wheelhouse \
       --no-build-isolation --no-deps /opt/flagrt/src/FlagGems \
    && ACCELERATOR=ascend ASCEND_KERNEL=1 CUDA_KERNEL=0 \
       FLAGGEMS_KERNEL=0 FLAGGEMS_PYTHON=1 FLAGOS_BUILD_JOBS=8 \
       python3 -m pip install --no-index --find-links=/opt/flagrt/wheelhouse \
       --no-build-isolation --no-deps /opt/flagrt/src/Torch-FL \
    && python3 /opt/flagrt/src/Torch-FL/scripts/patch_triton_ascend.py \
    && rm -rf /root/.cache/pip /root/.triton/cache /opt/flagrt/src

COPY verify_runtime.py /opt/flagrt/verify_runtime.py
COPY patch_triton_ascend_3_2_1.py /opt/flagrt/patch_triton_ascend_3_2_1.py

RUN python3 /opt/flagrt/patch_triton_ascend_3_2_1.py \
        --triton-path /usr/local/python3.11.15/lib/python3.11/site-packages/triton \
    && source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && make -C /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test \
        MPI_HOME=/usr/local/mpich ASCEND_DIR=/usr/local/Ascend/ascend-toolkit/latest \
    && python3 /opt/flagrt/verify_runtime.py --static

WORKDIR /workspace/FlagPerf/base
