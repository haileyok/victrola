# syntax=docker/dockerfile:1

# ── Stage 1: build the React/Vite frontend ──────────────────────────────────
FROM node:22-slim AS web-build

WORKDIR /build/web

# Install deps first (better layer caching)
COPY web/package.json web/package-lock.json ./
RUN npm ci

# Create the output directory so vite.config.ts (outDir: "../src/web/static")
# resolves correctly within the build container.
RUN mkdir -p /build/src/web/static

# Copy source and build — vite outputs to /build/src/web/static
COPY web/ ./
RUN npm run build

# ── Stage 2: Python runtime with Deno + Playwright Chromium ────────────────
FROM python:3.12-slim

# System deps:
#   - curl/unzip: Deno install
#   - ca-certificates: TLS for Deno/httpx
#   - libnss3, libnspr4, libatk1.0-0, libatk-bridge2.0-0, libcups2, libdrm2,
#     libxkbcommon0, libxcomposite1, libxdamage1, libxfixes3, libxrandr2,
#     libgbm1, libpango-1.0-0, libcairo2, libasound2, libatspi2.0-0:
#     Playwright Chromium runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        ca-certificates \
        gosu \
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
        libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Deno — pinned to a stable release
ARG DENO_VERSION=2.9.4
RUN curl -fsSL https://deno.land/install.sh | sh -s v${DENO_VERSION} \
    && mv /root/.deno/bin/deno /usr/local/bin/deno \
    && deno --version

# Install uv (fast Python package manager — matches the project's uv.lock)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY src/ ./src/
COPY main.py ./

# Copy pre-built frontend assets from Stage 1
COPY --from=web-build /build/src/web/static ./src/web/static

# Install Playwright Chromium browser (for web.save_url_as_pdf tool)
RUN uv run playwright install --with-deps chromium

# Create a non-root user for the application
RUN useradd --create-home --shell /bin/bash victrola \
    && mkdir -p /app/data \
    && chown -R victrola:victrola /app

# Entrypoint: fix data dir ownership on bind mounts, then drop to victrola user
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Persistent data: SQLite store, secrets.json, workspace, logs
VOLUME ["/app/data"]

# Default config — safe for containers
ENV WEB_HOST=0.0.0.0 \
    WEB_PORT=8000 \
    DATA_DIR=data \
    WORKSPACE_DIR=data/workspace \
    # Chromium can't use its namespace sandbox as non-root in a container
    WEB_PDF_CHROMIUM_NO_SANDBOX=1

EXPOSE 8000

# Lightweight healthcheck — the app opens port 8000 when uvicorn is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/api/status || exit 1

# Runs as root; entrypoint chowns the data dir then drops to victrola via gosu
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uv", "run", "python", "main.py", "serve"]
