# Milestone 2: LLM Fine-tuning with LoRA

## Overview
Fine-tuned Qwen/Qwen3-0.6B-Base model using LoRA (Low-Rank Adaptation) on 100 Alpaca examples with memory-efficient training.

## Training Configuration

### LoRA Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Rank (r) | 32 | Dimensionality of low-rank matrices |
| Alpha | 64 | Scaling factor (2×r) |
| Dropout | 0.2 | Regularization |
| Target Modules | 7 modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Batch Size | 4 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 16 |
| Learning Rate | 2e-5 |
| Optimizer | PagedAdamW 8-bit |
| Precision | FP16 (mixed precision) |

## Training Results

### Performance Metrics
- **Training Duration**: 9.72 seconds
- **Final Training Loss**: 1.5358
- **Training Samples**: 100 examples

### Parameter Efficiency
- **Total Parameters**: 616,235,008 (616.24M)
- **Trainable Parameters**: 20,185,088 (20.19M)
- **Trainable Percentage**: 3.28%

**Key Insight**: LoRA updates only 3.28% of model parameters, drastically reducing memory and compute requirements.

## Model Evaluation

Tested on diverse prompts not seen during training:

1. **Explanation Task**: Generated coherent AI definition with examples
2. **Creative Task**: Produced properly formatted haiku about the ocean
3. **Classification Task**: Correctly identified positive sentiment with reasoning

All test prompts demonstrated successful instruction-following behavior.

## Key Files Generated

### Adapter Weights (`adapters/`)
- `adapter_model.safetensors` (78 MB) - Trained LoRA adapter weights
- `adapter_config.json` - LoRA configuration
- Tokenizer files (vocab.json, merges.txt, tokenizer.json, etc.)

### Training Logs
- `training_log.txt` - Human-readable training summary
- `training_log.json` - Machine-readable training metrics

## Results
Successfully fine-tuned model using LoRA with efficient parameter usage and demonstrated instruction-following capability across multiple task types.
