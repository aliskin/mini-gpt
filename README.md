# MiniGPT with JAX and Flax NNX

## Motivation

This repository is a learning project for understanding how Transformer language models work in practice.

The starting point is the excellent DeepLearning.AI course [Build and Train an LLM with JAX](https://www.deeplearning.ai/courses/build-and-train-an-llm-with-jax/), which provided the original MiniGPT implementation and training setup used in this project.

I am extending that code incrementally to better understand individual Transformer components, training behaviour, and how architectural changes affect generated text.

## Current experiments

The main experiments so far compare progressively more complete Transformer blocks:

- multi-head causal self-attention with residual connections
- adding pre-LayerNorm
- adding the feed-forward network with GELU
- increasing the context length
- masking padded tokens in the training loss
- tracking next-token accuracy alongside loss

The experiments use the same prompts and similar training settings where possible so that differences in generation quality are easier to compare.

Plots, training curves, and example generations are available in [`Evaluation.ipynb`](./Evaluation.ipynb).

## Model

The current model is a small GPT-style autoregressive language model with:

- GPT-2 tokenization
- learned token and positional embeddings
- causal multi-head self-attention
- pre-LayerNorm Transformer blocks
- feed-forward networks with GELU
- residual connections
- next-token prediction objective

The implementation uses **JAX** and **Flax NNX**.

## Dataset

The model is trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a dataset of short synthetic stories designed for training and evaluating small language models.

Stories are tokenized with the GPT-2 tokenizer and padded or truncated to a fixed context length.

## Training

Training uses:

- **JAX** for computation
- **Flax NNX** for the model
- **Optax** for optimization
- warmup + cosine learning-rate decay
- masked cross-entropy loss
- next-token accuracy
- **Orbax** checkpoints

## Evaluation

The project currently focuses on simple comparisons between model variants using:

- training loss
- next-token accuracy
- generated text from a fixed set of prompts

See [`Evaluation.ipynb`](./Evaluation.ipynb) for plots and generation examples.

## Acknowledgements

The original implementation and training workflow are based on the DeepLearning.AI course:

**[Build and Train an LLM with JAX](https://www.deeplearning.ai/courses/build-and-train-an-llm-with-jax/)**

The course was extremely helpful as a practical introduction to implementing and training a small language model with JAX and Flax NNX.