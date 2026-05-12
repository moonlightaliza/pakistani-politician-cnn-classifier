# ── Stage 1: Base image ────────────────────────────────────────────────────────
# Using slim Python image (CPU-only inference; swap to nvidia/cuda image for GPU)
FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────────────────
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────────
COPY api/app.py .

# ── Copy model weights + class list ───────────────────────────────────────────
# These paths mirror your local project structure
COPY src/models/resnet/resnet50_politician.pth  src/models/resnet/resnet50_politician.pth
COPY src/models/resnet/classes.json             src/models/resnet/classes.json

# ── Environment variables (overridable at runtime) ─────────────────────────────
ENV MODEL_PATH=src/models/resnet/resnet50_politician.pth \
    CLASSES_PATH=src/models/resnet/classes.json \
    TOP_N=5 \
    PORT=5000

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Run with Gunicorn (production WSGI server) ─────────────────────────────────
# 2 workers × 2 threads = 4 concurrent requests; tune to your instance size
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "2", "--timeout", "120", "app:app"]gi