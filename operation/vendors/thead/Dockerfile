FROM flagtree-ppu-3.6-dev:202608 AS builder
ENV PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
COPY flaggems-source.tar /tmp/flaggems-source.tar
RUN echo 'cdca6e02ee2137c6dfa80ba446b3a709b130a2ef8eb901d906873255c6bba010  /tmp/flaggems-source.tar' | sha256sum -c - \
    && mkdir /tmp/flaggems \
    && tar xf /tmp/flaggems-source.tar -C /tmp/flaggems \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+g654e2a3a python -m pip wheel --no-deps --wheel-dir /wheels /tmp/flaggems \
    && python -m pip download --only-binary=:all: --no-deps --dest /wheels PyYAML==6.0.1 SQLAlchemy==2.0.48

FROM flagtree-ppu-3.6-dev:202608
LABEL flagperf.operation.vendor="thead" \
      flagperf.flaggems.commit="654e2a3a7956cc675886d71defbc1ea5aa6f69ba" \
      flagperf.runtime.release_stage="candidate"
COPY --from=builder /wheels /tmp/operation-wheels
RUN python -m pip install --no-index --no-deps /tmp/operation-wheels/*.whl \
    && rm -r /tmp/operation-wheels
