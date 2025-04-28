FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000
    

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn uvicorn[standard]

# Copy the application code
COPY . .

# Expose port
EXPOSE 8000

# Set a non-root user for security
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Resource limits appropriate for Render free tier (512MB RAM, 0.1 CPU)
ENV WORKERS=1 \
    WORKER_CLASS="uvicorn.workers.UvicornWorker" \
    TIMEOUT=60 \
    GRACEFUL_TIMEOUT=60 \
    KEEP_ALIVE=5

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import http.client; conn = http.client.HTTPConnection('localhost:${PORT}'); conn.request('GET', '/health/ready'); response = conn.getresponse(); exit(0) if response.status == 200 else exit(1)"

# Command to run the application
CMD gunicorn app.main:app \
    --workers ${WORKERS} \
    --worker-class ${WORKER_CLASS} \
    --bind 0.0.0.0:${PORT} \
    --timeout ${TIMEOUT} \
    --graceful-timeout ${GRACEFUL_TIMEOUT} \
    --keep-alive ${KEEP_ALIVE} \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --log-level info \
    --access-logfile - \
    --worker-tmp-dir /dev/shm