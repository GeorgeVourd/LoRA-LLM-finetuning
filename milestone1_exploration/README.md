# Milestone 1: Dataset Exploration and Model Setup

## Overview
This milestone explores the yahma/alpaca-cleaned dataset and sets up the Qwen/Qwen3-0.6B-Base model for fine-tuning.

## What's Inside

### 1. Dataset Analysis
- **Dataset**: yahma/alpaca-cleaned (51,760 examples)
- **Structure**: Each example contains `instruction`, `input`, and `output` fields
- **10 Sample Examples**: Displayed to understand format and content

### 2. Statistical Analysis
Computed descriptive statistics for text lengths:

| Metric | Instruction | Input | Output (Response) |
|--------|-------------|-------|-------------------|
| Mean | 10.54 words | 4.19 words | 109.94 words |
| Median | 10.00 words | 0.00 words | 80.00 words |
| Std Dev | 6.65 words | 11.71 words | 103.20 words |

**Key Insight**: Most examples have no input field (median = 0), instructions are concise (~10 words), and responses are detailed (~110 words).

### 3. Model Setup
- **Model**: Qwen/Qwen3-0.6B-Base
- **Parameters**: 596M total parameters
- **Memory**: ~1.11 GB (float16) / ~2.22 GB (float32)
- **Test Generation**: Successfully generated text with base model

### 4. Data Preprocessing
- **Format**: Applied Alpaca prompt template to all examples
- **Template Structure**:
  ```
  Below is an instruction that describes a task...
  ### Instruction: {instruction}
  ### Input: {input}
  ### Response: {output}
  ```
- **Tokenization**: Demonstrated tokenization process (avg ~182 tokens per example)

### 5. Training Subset Creation
- **Created**: 100-example subset from full dataset
- **Saved**: Raw subset and formatted subset for Milestone 2
- **Purpose**: Efficient training on limited resources

## Key Files Generated
- `subset_data/` - Raw 100-example subset
- `formatted_subset_data/` - Alpaca-formatted 100-example subset

## Results
All dataset exploration, model setup, and preprocessing tasks completed successfully.
