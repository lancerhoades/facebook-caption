FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg imagemagick fonts-dejavu-core wget curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Font for MoviePy TextClip (your code expects this path)
RUN mkdir -p /usr/local/share/fonts \
 && ln -s /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf /usr/local/share/fonts/MREARLN.TTF

WORKDIR /app

# Install Python deps first (cache-friendly)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY . /app

# Helpful for ImageMagick detection by moviepy
ENV IMAGEMAGICK_BINARY=convert
ENV PYTHONUNBUFFERED=1

# Start the RunPod worker
CMD ["python", "-u", "handler.py"]
