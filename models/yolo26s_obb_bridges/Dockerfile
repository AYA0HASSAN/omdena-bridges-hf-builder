# Builder stage: training + batch dependencies (GPU torch).
FROM ghcr.io/astral-sh/uv:0.11.8-python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy
ARG FAIR_PY_OPS_VERSION=0.3.17

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev gdal-bin libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    torch==2.10.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    "ultralytics>=8.4.0" \
    rasterio==1.5.1 \
    pyproj==3.7.2 \
    geopandas==1.1.4 \
    shapely==2.1.2 \
    Pillow==12.3.0 \
    onnx==1.22.0 \
    onnxscript==0.7.1 \
    "fair-py-ops[k8s,serve]==${FAIR_PY_OPS_VERSION}"

# Runtime stage: minimal training/batch image.
FROM python:3.13.13-slim-trixie AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgdal36 gdal-bin libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY models/yolo26s_obb_bridges models/yolo26s_obb_bridges
COPY models/conftest.py models/conftest.py
COPY models/test_integration.py models/test_integration.py

ENTRYPOINT ["/usr/local/bin/python"]

# Test stage: pytest extras for CI.
FROM runtime AS test
ARG FAIR_PY_OPS_VERSION=0.3.17
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "fair-py-ops[test]==${FAIR_PY_OPS_VERSION}"

# Inference builder: distroless-ready CPU venv (ONNX serving, no torch/ultralytics).
#
# The whole serving package - both __init__.py files, obb_utils.py, serving.py and pipeline.py - is
# copied below, not just pipeline.py: pipeline.py does `from .obb_utils import ...` and
# `from .serving import predict`, so a partial copy leaves the container unable to import it at start
# (ticket [BRIDGES-XX], agent feedback #858). pipeline.py stays in the image because
# .github/workflows/build-model-images.yml's smoke test imports `models.<name>.pipeline` generically
# across every model pack; it re-exports `predict` from `serving.py` rather than defining it, so that
# import still resolves.
#
# `serving.py` is the module that actually matters for isolation: it has no `zenml` import at all, so
# `MODEL_MODULE=models.yolo26s_obb_bridges.serving` (what the STAC item's `mlm:entrypoint` - and hence
# the real knative deployment - points at) imports cleanly without ever touching zenml. `pipeline.py`
# can't offer that guarantee: its `@step`/`@pipeline` decorators need `zenml` importable at module load,
# and fair/utils/model_validator.py's AST check requires those decorators to live in pipeline.py, so
# that file will always pull zenml in regardless of what gets copied here.
#
# `zenml` ends up installed in this venv either way, because it's an unconditional dependency of
# fair-py-ops (not gated behind the `[serve]` extra) - see "Known limitations" in README.md and the
# hotosm/fAIr-models follow-up issue proposing to move it behind a training extra. ultralytics/torch are
# genuinely absent (never installed here), which is what actually matters for image size and the AGPL
# ultralytics dependency not leaking into the served image.
FROM ghcr.io/astral-sh/uv:0.11.8-trixie-slim AS inference-builder
ENV UV_PYTHON_INSTALL_DIR=/python UV_PYTHON_PREFERENCE=only-managed
ARG FAIR_PY_OPS_VERSION=0.3.17
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.13
WORKDIR /app
RUN --mount=type=cache,target=/python/.cache/uv \
    uv venv && uv pip install \
    "fair-py-ops[serve]==${FAIR_PY_OPS_VERSION}" \
    onnxruntime==1.28.0 \
    rasterio==1.5.1 \
    pyproj==3.7.2 \
    shapely==2.1.2 \
    numpy==2.5.2 \
    Pillow==12.3.0
COPY models/__init__.py /app/models/__init__.py
COPY models/yolo26s_obb_bridges/__init__.py /app/models/yolo26s_obb_bridges/__init__.py
COPY models/yolo26s_obb_bridges/obb_utils.py /app/models/yolo26s_obb_bridges/obb_utils.py
COPY models/yolo26s_obb_bridges/serving.py /app/models/yolo26s_obb_bridges/serving.py
COPY models/yolo26s_obb_bridges/pipeline.py /app/models/yolo26s_obb_bridges/pipeline.py

# Inference stage: distroless live-serving runtime.
#
# fair-py-ops installs a `fair_zenml_patch.pth` that runs `import fair._patch_zenml` at *every*
# Python interpreter startup via site.py - unconditionally, regardless of what any script imports -
# to patch a ZenML server enum for Postgres support. That patch itself imports zenml, which is exactly
# what pulled zenml into sys.modules even though serving.py never imports it (E2's actual failure
# mode). It's irrelevant here: this container never talks to a ZenML server. FAIR_SKIP_ZENML_PATCH=1
# is the patch's own documented escape hatch (see fair/_patch_zenml.py) for exactly this case.
FROM gcr.io/distroless/python3-debian12:nonroot AS inference
WORKDIR /app
COPY --from=inference-builder /python /python
COPY --from=inference-builder /app/.venv /app/.venv
COPY --from=inference-builder /app/models /app/models
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app FAIR_SKIP_ZENML_PATCH=1
EXPOSE 8080
ENTRYPOINT ["/app/.venv/bin/python", "-m", "uvicorn"]
CMD ["fair.serve.base:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
