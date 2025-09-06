# Stable base (Debian 12/bookworm), avoids trixie churn
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    IMAGEMAGICK_BINARY=convert

# System deps (lean)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg imagemagick fonts-dejavu-core wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Font alias some code expects
RUN ln -s /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf /usr/local/share/fonts/MREARLN.TTF || true

WORKDIR /app

# Install Python deps first (cache-friendly)
COPY requirements.txt .
RUN python -m pip install --upgrade pip==24.2 setuptools==70.0.0 wheel==0.44.0 \
 && pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Start the RunPod worker
CMD ["python", "-u", "handler.py"]
