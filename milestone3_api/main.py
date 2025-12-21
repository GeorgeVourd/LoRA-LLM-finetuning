"""
FastAPI service for serving the fine-tuned Qwen model with LoRA adapters.

This API provides endpoints for:
- Text generation using the fine-tuned model
- Health checks for service monitoring
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="The input prompt for text generation", min_length=1)
    max_new_tokens: Optional[int] = Field(default=100, description="Maximum number of tokens to generate", ge=1, le=512)
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature", ge=0.1, le=2.0)
    top_p: Optional[float] = Field(default=0.9, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    do_sample: Optional[bool] = Field(default=True, description="Whether to use sampling")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str = Field(..., description="The generated text")
    prompt: str = Field(..., description="The original prompt")
    num_tokens: int = Field(..., description="Number of tokens generated")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Device being used (cpu/cuda)")


def load_model_and_tokenizer():
    """
    Load the base model, apply LoRA adapters, and load the tokenizer.
    """
    global model, tokenizer, device

    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Get paths from environment variables or use defaults
        base_model_name = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen3-0.6B-Base")
        adapter_path = os.getenv("ADAPTER_PATH", "../milestone2_finetuning/adapters")

        logger.info(f"Loading base model: {base_model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        logger.info("✓ Tokenizer loaded")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )

        if device == "cpu":
            base_model = base_model.to(device)

        logger.info("✓ Base model loaded")

        # Check if adapter path exists
        if os.path.exists(adapter_path):
            logger.info(f"Loading LoRA adapters from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            logger.info("✓ LoRA adapters loaded")
        else:
            logger.warning(f"Adapter path not found: {adapter_path}")
            logger.warning("Using base model without LoRA adapters")
            model = base_model

        # Set model to evaluation mode
        model.eval()
        logger.info("✓ Model ready for inference")

        return True

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads the model on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Starting up API service...")
    try:
        load_model_and_tokenizer()
        logger.info("API service ready!")
    except Exception as e:
        logger.error(f"Failed to start API service: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down API service...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Qwen Fine-tuned Model API",
    description="REST API for text generation using fine-tuned Qwen model with LoRA",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Qwen Fine-tuned Model API",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "POST - Generate text from a prompt",
            "/health": "GET - Check service health"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns service status and model availability.
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=device if device is not None else "unknown"
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text based on the provided prompt.

    Args:
        request: GenerateRequest containing the prompt and generation parameters

    Returns:
        GenerateResponse containing the generated text

    Raises:
        HTTPException: If model is not loaded or generation fails
    """
    # Check if model is loaded
    if model is None or tokenizer is None:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is not ready."
        )

    try:
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")

        # Format prompt in Alpaca instruction style (matching training format)
        formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{request.prompt}

### Response:
"""

        # Tokenize the formatted input
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        input_length = inputs['input_ids'].shape[1]

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.do_sample,
                top_p=request.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode generated text
        full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part (after "### Response:")
        if "### Response:" in full_generated_text:
            generated_text = full_generated_text.split("### Response:")[-1].strip()
        else:
            # Fallback: remove the formatted prompt
            generated_text = full_generated_text[len(formatted_prompt):].strip()

        # Calculate number of new tokens generated
        num_tokens_generated = outputs[0].shape[0] - input_length

        logger.info(f"Generated {num_tokens_generated} tokens")

        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            num_tokens=num_tokens_generated
        )

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during text generation: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
