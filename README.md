# MiniGPT with JAX and Flax NNX

## Motivation

This repository is a learning project for understanding how Transformer language models work in practice.

The starting point is the excellent DeepLearning.AI course [Build and Train an LLM with JAX](https://www.deeplearning.ai/courses/build-and-train-an-llm-with-jax/), which provided the original MiniGPT implementation and training setup used in this project.

I am extending that code incrementally to better understand individual Transformer components, training behaviour, and how architectural changes affect generated text.

## Current experiments

The next controlled comparison is described in [EXPERIMENTS.md](./EXPERIMENTS.md):
attention with pre-LayerNorm, adding an FFN, and adding final LayerNorm.
Each run has a config file that can be passed directly to `train.py --config`.

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

10% of the loaded stories are reserved for validation. Identical story texts are
removed before splitting to avoid duplicate leakage. Generating training and validation 
datasets uses the following parameters: `--max-stories` (the total number of stories 
before splitting), `--validation-fraction` (a percentage of data used for validation), 
and `--split-seed`. With 25K distinct stories, the default split gives 22,500 training 
and 2,500 validation stories.

After each epoch, the model is evaluated on the entire validation split, including
the last partial batch. Padding and positions without a real next-token target are 
excluded throughout.

- **Loss**: mean next-token cross-entropy in natural log units.
- **Accuracy**: fraction of valid targets matched by the highest-logit token.
- **Perplexity**: `exp(loss)`; lower is better. For example, loss 2 corresponds
  to perplexity about 7.39.

Both loss and accuracy are weighted by valid token counts across batches.
Perplexity is calculated after aggregating loss, rather than averaging batch
perplexities. Training metrics are measured as weights change during the epoch;
validation metrics describe the model at the end of that epoch.

For example, run the current FFN model with:

```sh
python train.py --name ffn_validation_25k_256 \
  --data tinystories_data/TinyStories-25000.txt \
  --max-stories 25000 --maxlen 256 --epochs 3 \
  --validation-fraction 0.1 --split-seed 123 --seed 42
```

Use a new experiment name for each run. `metrics.json` retains training-window
records (`loss`, `accuracy`, `perplexity`, `tokens`, and `learning_rate`).
`epoch_metrics.json` records `train_loss`, `train_accuracy`, `train_perplexity`,
`val_loss`, `val_accuracy`, `val_perplexity`, and token counts after every epoch.
`split.json` records the split sizes and settings.

Check `Evaluation.ipynb` to see a comparison of the three architectures using epoch metrics
as a quantitative check as well as generated texts from fixed prompts as a qualitative check. 

## Acknowledgements

The original implementation and training workflow are based on the DeepLearning.AI course:

**[Build and Train an LLM with JAX](https://www.deeplearning.ai/courses/build-and-train-an-llm-with-jax/)**

The course was extremely helpful as a practical introduction to implementing and training a small language model with JAX and Flax NNX.

Special thanks to CodeX and my first experience with vibe coding :)
