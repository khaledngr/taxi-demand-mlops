# The serving image: code + locked environment, no model inside. The model is resolved at start-up
# from the registry (models:/taxi-demand@production), so promotion and rollback never rebuild the image.
#   docker build -t taxi-api:$(git rev-parse --short HEAD) .
#   make docker-serve
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# dependencies first: this layer is cached until uv.lock changes
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

# then the code, which changes often
COPY taxi/ taxi/
COPY app/ app/
COPY params.yaml policy.yaml ./

ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=$GIT_COMMIT
ENV GIT_COMMIT=$GIT_COMMIT
RUN useradd --create-home api
USER api
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
