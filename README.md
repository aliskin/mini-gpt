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

Training now reserves 10% of the loaded stories for validation. The split uses
`--split-seed 123`, independently of the model seed. Identical story texts are
removed before splitting to avoid duplicate leakage. Use the same input file,
`--max-stories`, `--validation-fraction`, and `--split-seed` for every comparison.
With 25K distinct stories, the default split gives 22,500 training and 2,500
validation stories; `--max-stories` describes the total before splitting.

After each epoch the model is evaluated on the entire validation split, including
the last partial batch. Validation performs no gradient or optimizer updates.
Padding and positions without a real next-token target are excluded throughout.

- **Loss**: mean next-token cross-entropy in natural log units; lower is better.
- **Accuracy**: fraction of valid targets matched by the highest-logit token;
  higher is better.
- **Perplexity**: `exp(loss)`; lower is better. For example, loss 2 corresponds
  to perplexity about 7.39. It is a transformation of loss, not an independent
  signal. Compare it using the same tokenizer, context length, and held-out data.

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

`Evaluation.ipynb` includes a comparison cell for these epoch metrics. Generated
text from fixed prompts remains useful as a qualitative check. Old training
results used all stories, so rerun baselines with the new split for a fair
comparison. An experiment name is only a label: use the variant config with `--config`
to select its architecture. Evaluation reconstructs each model from its saved
configuration and checkpoint metadata.

The training loop now consumes exactly one pass through training data per epoch;
previously the loader and the outer loop both requested multiple epochs.

Run the metric checks in your project environment with:

```sh
python -m unittest discover -s tests -v
```

## Acknowledgements

The original implementation and training workflow are based on the DeepLearning.AI course:

**[Build and Train an LLM with JAX](https://www.deeplearning.ai/courses/build-and-train-an-llm-with-jax/)**

The course was extremely helpful as a practical introduction to implementing and training a small language model with JAX and Flax NNX.