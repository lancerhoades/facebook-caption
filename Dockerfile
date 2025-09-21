# Stable base (Debian 12/bookworm)
FROM python:3.11-slim-bookworm
ARG BUILD_REV=.time

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    IMAGEMAGICK_BINARY=/usr/bin/convert

# [STEP 1] OS deps (ffmpeg has libass); libass9 just to be explicit; fontconfig + fallback fonts
RUN set -eux; \
    echo "[STEP 1] apt-get install"; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        gawk ffmpeg libass9 fontconfig fonts-dejavu-core imagemagick ca-certificates; \
    ln -sf /usr/bin/convert /usr/local/bin/convert; \
    rm -rf /var/lib/apt/lists/*; \
    echo "[STEP 1 DONE]"

WORKDIR /app

# [STEP 2] Python deps (print progress around pip)
COPY requirements.txt .
RUN set -eux; \
    echo "[STEP 2] pip upgrade"; \
    python -m pip install --upgrade pip==24.2 setuptools==70.0.0 wheel==0.44.0; \
    echo "[STEP 2a] pip install -r requirements.txt (prefer wheels)"; \
    pip install --only-binary=:all: --no-cache-dir -r requirements.txt || \
    (echo "[STEP 2a fallback] retry without wheels-only" && pip install --no-cache-dir -r requirements.txt); \
    echo "[STEP 2 DONE]"

# [STEP 3] App code
COPY caption.py handler.py ./
COPY tools/ /app/tools/

# [STEP 4] Fonts and cache
COPY fonts/ /usr/local/share/fonts/custom/
RUN set -eux; echo "[STEP 4] fc-cache"; fc-cache -f -v || true; echo "[STEP 4 DONE]"

# [STEP 5] Entrypoint
ENTRYPOINT ["python", "-u", "handler.py"]
