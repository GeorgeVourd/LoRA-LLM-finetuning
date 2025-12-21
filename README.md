# LoRA LLM Fine-tuning

Fine-tuning Qwen3-0.6B-Base model using LoRA on the Alpaca-cleaned dataset. Includes Jupyter notebook exploration, training with Hugging Face Transformers and PEFT, and FastAPI deployment with Docker containerization.

## Overview

This project demonstrates an end-to-end pipeline for fine-tuning large language models using LoRA (Low-Rank Adaptation), from dataset exploration to production deployment.

**Key Technologies:**
- Model: Qwen/Qwen3-0.6B-Base (596M parameters)
- Dataset: yahma/alpaca-cleaned (51,760 examples)
- Fine-tuning: LoRA with PEFT library
- Training Environment: Google Colab with NVIDIA A100 GPU
- Deployment: FastAPI + Docker

## What's Inside This Project

This project is organized into three milestones, each with its own detailed README:

### **Milestone 1: Dataset Exploration** (`milestone1_exploration/`)
Dataset analysis, preprocessing, and model setup.

- Explores the yahma/alpaca-cleaned dataset (51,760 examples)
- Analyzes text statistics and data structure
- Loads and tests Qwen/Qwen3-0.6B-Base model
- Applies Alpaca prompt template formatting
- Creates a 100-example training subset
- Executed on Google Colab with NVIDIA A100 GPU

**📄 See [`milestone1_exploration/README.md`](milestone1_exploration/README.md) for detailed documentation**

### **Milestone 2: Model Fine-tuning** (`milestone2_finetuning/`)
Fine-tuning the model using LoRA with memory-efficient training.

- Fine-tunes Qwen3-0.6B using LoRA (rank=32, alpha=64)
- Trains on 100 Alpaca examples with efficient parameters
- Uses PagedAdamW 8-bit optimizer and FP16 precision
- Achieves 3.28% trainable parameters (20.19M / 616.24M total)
- Generates lightweight adapter weights (78 MB)
- Trained on Google Colab with NVIDIA A100 GPU

**📄 See [`milestone2_finetuning/README.md`](milestone2_finetuning/README.md) for detailed documentation**

### **Milestone 3: API Deployment** (`milestone3_api/`)
Production-ready FastAPI service with Docker deployment.

- FastAPI web service for model inference
- Docker containerization for easy deployment
- RESTful API endpoints for text generation
- Testing suite included

**📄 See [`milestone3_api/README.md`](milestone3_api/README.md) for detailed documentation**

## Project Structure

```
├── milestone1_exploration/
│   ├── README.md                    
│   ├── milestone1_exploration.ipynb 
│   ├── subset_data/                 # Raw 100-example subset
│   └── formatted_subset_data/       # Formatted training data
│
├── milestone2_finetuning/
│   ├── README.md                    
│   ├── milestone2_finetuning.ipynb  
│   ├── adapters/                    # LoRA adapter
│   ├── training_log.txt             # Training summary
│   └── training_log.json            # Training metrics
│
└── milestone3_api/
    ├── README.md                    
    ├── main.py                      # FastAPI application
    ├── test_api.py                  # API tests
    ├── Dockerfile                   # Container configuration
    ├── docker-compose.yml           # Docker orchestration
    ├── requirements.txt             
    └── adapters/                    # LoRA adapters
```

## Key Results

### Training Performance (on NVIDIA A100 GPU)
- **Training Duration**: ~10 seconds
- **Final Loss**: 1.5358
- **Trainable Parameters**: 3.28% (20.19M / 616.24M)
- **Adapter Size**: 78 MB

### Model Capabilities
Successfully demonstrates instruction-following on:
- Explanation tasks
- Creative writing (e.g., haiku generation)
- Classification tasks (e.g, sentiment analysis of text)
- General question-answering

## Technologies Used

- **Hugging Face Transformers**: Model loading and inference
- **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA implementation
- **bitsandbytes**: 8-bit optimizer for memory efficiency
- **Google Colab**: Cloud notebook environment with A100 GPU
- **FastAPI**: Web API framework
- **Docker**: Containerization and deployment
- **Jupyter**: Interactive development and exploration

## Acknowledgments

- **Model**: Qwen/Qwen3-0.6B-Base by Alibaba Cloud
- **Dataset**: yahma/alpaca-cleaned (Stanford Alpaca cleaned version)
- **LoRA**: Low-Rank Adaptation technique by Microsoft Research
- **Training Infrastructure**: Google Colab with NVIDIA A100 GPU
