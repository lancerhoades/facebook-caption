FROM python:3.11-slim

# System deps: ffmpeg for audio/video, imagemagick for MoviePy TextClip "caption" method, curl for uploading
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg imagemagick curl fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Provide a font at the exact path your script expects
# Your caption.py uses: /usr/local/share/fonts/MREARLN.TTF
# We'll symlink DejaVuSans-Bold there so it doesn't break.
RUN mkdir -p /usr/local/share/fonts && \
    ln -s /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf /usr/local/share/fonts/MREARLN.TTF

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy code
WORKDIR /app
COPY . /app

# RunPod serverless will start via handler.py
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "handler.py"]
