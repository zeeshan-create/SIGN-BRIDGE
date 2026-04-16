# SignBridge AI - Production Dockerfile for Railway
# ==================================================
# Key optimizations:
#   - Uses requirements.prod.txt (tflite-runtime vs full tensorflow = -2GB!)
#   - opencv-python-headless (no GUI libs = -500MB!)
#   - Multi-stage build (no build tools in final image)
#   - Removed unused packages: PyAudio, pygame, matplotlib, etc.
#
# Estimated final size: ~1.5-2GB (down from 8.8GB)

# Stage 1: Builder - Compile Python packages
FROM python:3.11-slim as builder

# Install build dependencies (only for compiling Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install production requirements
COPY requirements.prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel && \
    pip install --no-cache-dir -r requirements.prod.txt

# Stage 2: Runtime - Minimal image with compiled packages
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/venv/lib/python3.11/site-packages"

# Install ONLY runtime system dependencies (no -dev packages, no audio libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libfontconfig1 \
    libice6 \
    libx11-6 \
    libharfbuzz0b \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY main.py .
COPY src/ ./src/
COPY models/ ./models/
COPY users.json .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 5000

# Health check (simplified - no requests library needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 5000)); s.close()" 2>/dev/null || exit 1

# Production command
CMD exec gunicorn --worker-class eventlet \
    --workers 1 \
    --bind 0.0.0.0:$PORT \
    --timeout 120 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile - \
    "main:app"
