# Multi-stage build for Contract Correspondence Classification API
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements_production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_production.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/embeddings data/models data/backups data/processed

# Set permissions
RUN chmod +x start_production.py

# Create non-root user for security
RUN adduser --disabled-password --gecos '' --uid 1000 apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "start_production.py"]

# Production image
FROM base as production

# Set production environment
ENV ENVIRONMENT=production
ENV WORKERS=4
ENV PORT=8000

# Use gunicorn for production
CMD ["gunicorn", "production_api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Development image
FROM base as development

ENV ENVIRONMENT=development
CMD ["python", "start_production.py"]