# Milestone 3: API and Containerization

A production-ready Docker containerized REST API for the fine-tuned Qwen model with LoRA adapters.

## Quick Start with Docker

### Prerequisites

- Docker installed (version 20.10+)
- 8+ GB RAM available
- 10 GB free disk space
- Completed Milestone 2 (LoRA adapters trained)

## Step-by-Step Deployment Guide

### Step 1: Verify LoRA Adapters

The LoRA adapters from Milestone 2 should be present in the `adapters/` directory.

Verify the adapters exist:

```bash
ls adapters/
# Should show: adapter_model.safetensors, adapter_config.json, tokenizer files
```

If adapters are missing, copy them from Milestone 2:

```bash
# From the milestone3_api directory
mkdir -p adapters
cp -r ../milestone2_finetuning/adapters/* adapters/

# Or on Windows:
# mkdir adapters
# xcopy /E /I ..\milestone2_finetuning\adapters adapters
```

### Step 2: Build the Docker Image

```bash
# Build the image (this takes 3-5 minutes)
docker build -t qwen-api .
```

**Expected output:**
```
[+] Building 180.5s (15/15) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 1.23kB
 ...
 => exporting to image
 => => naming to docker.io/library/qwen-api
```

### Step 3: Run the Container

```bash
# Run in detached mode (background)
docker run -d -p 8000:8000 --name qwen-api-container qwen-api
```

**With GPU support (optional):**
```bash
docker run -d -p 8000:8000 --gpus all --name qwen-api-container qwen-api
```

### Step 4: Wait for Model to Load

The model takes 30-60 seconds to load on first start. Check the logs:

```bash
# Follow the logs
docker logs -f qwen-api-container

# Look for this message:
# INFO:     API service ready!
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

Press `Ctrl+C` to stop following logs (container keeps running).

### Step 5: Test the API

#### Test 1: Health Check

```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### Test 2: Simple Text Generation

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "max_new_tokens": 50,
    "temperature": 0.7
  }'
```

**Expected response:**
```json
{
  "generated_text": "Artificial Intelligence (AI) refers to the development",
  "prompt": "What is AI?",
  "num_tokens": 50
}
```

#### Test 3: With Alpaca Format (Recommended)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain artificial intelligence in simple terms.",
    "max_new_tokens": 150,
    "temperature": 0.7
  }'
```

#### Test 4: Creative Generation

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short poem about coding.",
    "max_new_tokens": 200,
    "temperature": 0.9,
    "top_p": 0.95
  }'
```

## Using Docker Compose (Alternative Method)

Docker Compose makes it even easier:

### Start the Service

```bash
# Start in background
docker-compose up -d

# Or start and view logs
docker-compose up
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?", "max_new_tokens": 50}'
```

### View Logs

```bash
docker-compose logs -f
```

### Stop the Service

```bash
docker-compose down
```

## API Endpoints Reference

### 1. Root Endpoint

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "Qwen Fine-tuned Model API",
  "version": "1.0.0",
  "endpoints": {
    "/generate": "POST - Generate text from a prompt",
    "/health": "GET - Check service health"
  }
}
```

### 2. Health Check

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### 3. Generate Text

**Request:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "YOUR_PROMPT_HERE",
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": true
  }'
```

**Parameters:**
- `prompt` (required): Text prompt for generation
- `max_new_tokens` (optional): Number of tokens to generate (1-512, default: 100)
- `temperature` (optional): Sampling temperature (0.1-2.0, default: 0.7)
- `top_p` (optional): Nucleus sampling (0.0-1.0, default: 0.9)
- `do_sample` (optional): Use sampling vs greedy (default: true)

**Response:**
```json
{
  "generated_text": "The generated response text...",
  "prompt": "YOUR_PROMPT_HERE",
  "num_tokens": 87
}
```

## Managing the Container

### View Running Containers

```bash
docker ps
```

### Stop the Container

```bash
docker stop qwen-api-container
```

### Start the Container Again

```bash
docker start qwen-api-container
```

### Remove the Container

```bash
docker stop qwen-api-container
docker rm qwen-api-container
```

### View Logs

```bash
# View all logs
docker logs qwen-api-container

# Follow logs in real-time
docker logs -f qwen-api-container

# View last 50 lines
docker logs --tail 50 qwen-api-container
```

### Access Container Shell (for debugging)

```bash
docker exec -it qwen-api-container /bin/bash
```

## Interactive API Documentation

Once the API is running, visit these URLs in your browser:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These provide interactive documentation where you can test the API directly.

## Python Test Script

Use the included test script:

```bash
# Install requests if needed
pip install requests

# Run tests
python test_api.py

# Test different URL
python test_api.py --url http://localhost:8000
```

## Troubleshooting

### Issue: "Connection refused" error

**Cause:** Container not running or model still loading

**Solution:**
```bash
# Check container status
docker ps -a

# Check logs for errors
docker logs qwen-api-container

# Wait 30-60 seconds for model to load
```

### Issue: Container exits immediately

**Cause:** Missing adapters or build error

**Solution:**
```bash
# Check if adapters exist
ls adapters/

# View container logs
docker logs qwen-api-container

# Rebuild without cache
docker build --no-cache -t qwen-api .
```

### Issue: 503 Error "Model not loaded"

**Cause:** Model still loading or failed to load

**Solution:**
```bash
# Wait 60 seconds, then check health
sleep 60
curl http://localhost:8000/health

# Check logs for loading progress
docker logs qwen-api-container
```

### Issue: Port 8000 already in use

**Solution:**
```bash
# Use different port
docker run -d -p 9000:8000 --name qwen-api-container qwen-api

# Then access via http://localhost:9000
```

### Issue: Out of memory

**Solution:**
```bash
# Check Docker memory settings (Docker Desktop > Settings > Resources)
# Ensure at least 8 GB allocated

# Or reduce concurrent requests
```

## Performance Notes

### Response Times (100 tokens)

| Hardware | First Request | Subsequent Requests |
|----------|--------------|---------------------|
| CPU (4 cores) | 15-30s | 10-25s |
| GPU (4GB VRAM) | 2-5s | 1-3s |
| GPU (8GB+ VRAM) | 1-3s | 0.5-2s |

### Memory Usage

- **CPU Mode:** 4-6 GB RAM
- **GPU Mode:** 2-4 GB RAM + 2-3 GB VRAM

## Environment Variables

Configure the API by setting environment variables:

```bash
docker run -d -p 8000:8000 \
  -e BASE_MODEL_NAME="Qwen/Qwen3-0.6B-Base" \
  -e ADAPTER_PATH="/app/adapters" \
  --name qwen-api-container \
  qwen-api
```

Available variables:
- `BASE_MODEL_NAME`: Model to load (default: Qwen/Qwen3-0.6B-Base)
- `ADAPTER_PATH`: Path to adapters (default: /app/adapters)
- `PORT`: API port (default: 8000)
- `HOST`: API host (default: 0.0.0.0)

## Production Deployment

For production use:

1. **Add authentication** (API keys, OAuth)
2. **Implement rate limiting**
3. **Use HTTPS** (reverse proxy with nginx/traefik)
4. **Add monitoring** (Prometheus, Grafana)
5. **Set up logging** (ELK stack, CloudWatch)
6. **Use orchestration** (Kubernetes, Docker Swarm)

## Clean Up

Remove everything when done:

```bash
# Stop and remove container
docker stop qwen-api-container
docker rm qwen-api-container

# Remove image
docker rmi qwen-api

# Or use docker-compose
docker-compose down --rmi all
```

## Files Included

- `main.py` - FastAPI application
- `Dockerfile` - Multi-stage Docker build
- `docker-compose.yml` - Easy deployment
- `requirements.txt` - Python dependencies
- `test_api.py` - Test script
- `adapters/` - LoRA adapter files (copied from Milestone 2)

