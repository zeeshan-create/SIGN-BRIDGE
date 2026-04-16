# SignBridge AI - Production Dockerfile for Fly.io
# =================================================
# Optimized for Fly.io deployment
# Full TensorFlow included for TFLite model compatibility

FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    PYTHONFAULTHANDLER=1

# Install minimal runtime dependencies
# libgl1 = OpenCV requirement
# libgomp1 = OpenMP for numpy/medpiapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.prod.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.prod.txt && \
    pip install --no-cache-dir mediapipe==0.10.11 && \
    pip cache purge && \
    rm -rf /root/.cache/pip/*

# Copy application code
COPY main.py .
COPY src/ ./src/
COPY models/ ./models/
COPY users.json .

# Create uploads directory
RUN mkdir -p uploads

# Health check using built-in socket (no extra deps)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('localhost',5000)); s.close()" || exit 1

EXPOSE 5000

# Production command
CMD exec gunicorn --worker-class eventlet \
    --workers 1 \
    --bind 0.0.0.0:$PORT \
    --timeout 120 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile - \
    "main:app"