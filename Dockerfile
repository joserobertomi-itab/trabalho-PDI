# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

ARG PDISEG_EXTRAS=

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "${PDISEG_EXTRAS}" ]; then \
        uv sync --frozen --no-install-project --no-dev --extra "${PDISEG_EXTRAS}"; \
    else \
        uv sync --frozen --no-install-project --no-dev; \
    fi

COPY src/pdiseg/ src/pdiseg/

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "${PDISEG_EXTRAS}" ]; then \
        uv sync --frozen --no-dev --extra "${PDISEG_EXTRAS}"; \
    else \
        uv sync --frozen --no-dev; \
    fi

FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="pdiseg" \
      org.opencontainers.image.description="Poultry packaging name-label segmentation and recognition (classical PDI)" \
      org.opencontainers.image.source="https://github.com/joserobertomi-itab/trabalho-PDI"

RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 pdiseg \
    && useradd --uid 1000 --gid pdiseg --create-home --shell /usr/sbin/nologin pdiseg

WORKDIR /app

COPY --from=builder --chown=pdiseg:pdiseg /app/.venv /app/.venv
COPY --chown=pdiseg:pdiseg src/pdiseg/ src/pdiseg/
COPY --chown=pdiseg:pdiseg pyproject.toml README.md CONTEXT.md ./
COPY --chown=pdiseg:pdiseg templates/ templates/
COPY --chown=pdiseg:pdiseg scripts/build-templates.py scripts/build-t2-report.py scripts/build-t2-simplified.py scripts/
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /data/input /data/output /data/calibration /data/templates /data/report \
    && chown -R pdiseg:pdiseg /data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    APP_USER=pdiseg \
    APP_UID=1000 \
    APP_GID=1000

USER root

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["pdiseg", "/data/input", "/data/output"]
