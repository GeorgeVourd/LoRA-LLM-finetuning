# Multi-stage Dockerfile for Qwen Fine-tuned Model API
# This Dockerfile creates a lean, production-ready container for the API service

# Stage 1: Base image with Python and system dependencies
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Stage 3: Final production image
FROM base as production

# Copy installed dependencies from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY main.py .

# Create directory for adapters
RUN mkdir -p /app/adapters

# Copy LoRA adapters (these should be copied from milestone2_finetuning/adapters)
# Note: When building, make sure to have the adapters in the build context
COPY adapters/ /app/adapters/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BASE_MODEL_NAME="Qwen/Qwen3-0.6B-Base"
ENV ADAPTER_PATH="/app/adapters"
ENV PORT=8000
ENV HOST="0.0.0.0"

# Expose the API port
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
