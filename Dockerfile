# Stable base (Debian 12/bookworm)
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    IMAGEMAGICK_BINARY=/usr/bin/convert

# System deps: ffmpeg (libass included), fontconfig for fc-cache, ImageMagick, certs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fontconfig imagemagick ca-certificates \
 && ln -sf /usr/bin/convert /usr/local/bin/convert \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first
COPY requirements.txt .
RUN python -m pip install --upgrade pip==24.2 setuptools==70.0.0 wheel==0.44.0 \
 && pip install --no-cache-dir -r requirements.txt

# App code
COPY caption.py handler.py ./

# Fonts: copy all and rebuild cache
COPY fonts/ /usr/local/share/fonts/custom/
RUN fc-cache -f -v || true

# Entrypoint for RunPod
ENTRYPOINT ["python", "-u", "handler.py"]
