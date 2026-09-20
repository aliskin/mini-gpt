import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import tiktoken

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax

from model import ARCHITECTURE_DEFAULTS, model_from_config
from story_dataset import create_dataloader, load_stories_from_file, split_stories


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

prep_target_batch = jax.vmap(
    lambda tokens: jnp.concatenate(
        (tokens[1:], jnp.array([0], dtype=tokens.dtype))
    )
)


def prepare_batch(batch):
    """
    Convert a Grain batch into JAX arrays.

    Grain currently returns tokens with shape:
        (seq_len, batch_size)

    MiniGPT expects:
        (batch_size, seq_len)
    """
    inputs = jnp.asarray(
        batch["tokens"],
        dtype=jnp.int32
    ).T

    targets = prep_target_batch(inputs)

    lengths = jnp.asarray(
        batch["seq_len"],
        dtype=jnp.int32
    )

    return inputs, targets, lengths


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

def loss_fn(model, batch):
    inputs, targets, lengths = batch
    logits = model(inputs)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    positions = jnp.arange(targets.shape[1])[None, :]
    loss_mask = positions < (lengths[:, None] - 1)

    # Return sums so differently sized batches/stories are weighted by tokens.
    loss_sum = jnp.where(loss_mask, token_losses, 0.0).sum()
    token_count = loss_mask.sum()
    correct_sum = ((jnp.argmax(logits, axis=-1) == targets) & loss_mask).sum()
    loss = loss_sum / jnp.maximum(token_count, 1)
    return loss, (loss_sum, correct_sum, token_count)


@nnx.jit
def train_step(model, optimizer, batch):
    (_, totals), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    optimizer.update(grads)
    return totals


@nnx.jit
def eval_step(model, batch):
    # Forward pass only: no gradient calculation or optimizer update.
    _, totals = loss_fn(model, batch)
    return totals


def summarize_metrics(loss_sum, correct_sum, token_count):
    if token_count == 0:
        raise ValueError("No valid next-token targets in these batches")
    loss = loss_sum / token_count
    return {
        "loss": loss,
        "accuracy": correct_sum / token_count,
        "perplexity": math.exp(loss),
        "tokens": int(token_count),
    }


def evaluate(model, dataloader):
    totals = [0.0, 0.0, 0.0]
    for batch in dataloader:
        batch_totals = eval_step(model, prepare_batch(batch))
        totals = [a + float(b) for a, b in zip(totals, batch_totals)]
    return summarize_metrics(*totals)


# ---------------------------------------------------------------------
# Checkpoint / logging helpers
# ---------------------------------------------------------------------

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_checkpoint(model, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(
        checkpoint_dir,
        nnx.state(model),
        force=True
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train(config):
    config = {**ARCHITECTURE_DEFAULTS, "tokenizer": "gpt2", **config}
    tokenizer = tiktoken.get_encoding(config["tokenizer"])
    config["vocab_size"] = tokenizer.n_vocab
    # ----- experiment directory -----

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = (
        config["name"]
        or f"{timestamp}_{config['max_stories']}stories"
    )

    experiment_dir = Path("experiments") / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_json(config, experiment_dir / "config.json")

    print(f"Experiment: {experiment_name}")
    print(f"Output:     {experiment_dir}")
    print()


    # ----- dataset -----
    stories = load_stories_from_file(config["data"], config["max_stories"])
    train_stories, validation_stories = split_stories(
        stories, config["validation_fraction"], config["split_seed"],
    )
    batches_per_epoch = len(train_stories) // config["batch_size"]
    if batches_per_epoch == 0:
        raise ValueError("Training split must contain at least one full batch")
    validation_dl = create_dataloader(
        validation_stories, tokenizer, config["batch_size"], config["maxlen"],
        drop_remainder=False,
    )
    save_json({
        "train_stories": len(train_stories),
        "validation_stories": len(validation_stories),
        "split_seed": config["split_seed"],
        "validation_fraction": config["validation_fraction"],
    }, experiment_dir / "split.json")
    print(f"Train: {len(train_stories):,}; validation: {len(validation_stories):,} stories")


    # ----- model -----
    model = model_from_config(config)


    # ----- optimizer -----

    total_steps = batches_per_epoch * config["epochs"]

    warmup_steps = max(
        1,
        int(total_steps * config["warmup_fraction"])
    )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config["learning_rate"],
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=config["end_learning_rate"],
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=config["weight_decay"],
        )
    )


    # ----- train -----

    metrics_history = []
    epoch_history = []
    global_step = 0

    for epoch in range(config["epochs"]):
        # Exactly one data epoch, with a reproducible new shuffle each time.
        text_dl = create_dataloader(
            train_stories, tokenizer, config["batch_size"], config["maxlen"],
            shuffle=True, seed=config["seed"] + epoch,
        )
        window_totals = [0.0, 0.0, 0.0]
        epoch_totals = [0.0, 0.0, 0.0]
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        for batch_index, batch in enumerate(text_dl, start=1):
            totals = train_step(model, optimizer, prepare_batch(batch))
            totals = [float(value) for value in totals]
            global_step += 1
            window_totals = [a + b for a, b in zip(window_totals, totals)]
            epoch_totals = [a + b for a, b in zip(epoch_totals, totals)]

            if global_step % config["log_every"] == 0 or batch_index == batches_per_epoch:
                record = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    **summarize_metrics(*window_totals),
                    "learning_rate": float(lr_schedule(global_step - 1)),
                }
                metrics_history.append(record)
                print(
                    f"step={global_step:5d} loss={record['loss']:.4f} "
                    f"accuracy={record['accuracy']:.4f} "
                    f"perplexity={record['perplexity']:.2f}"
                )
                window_totals = [0.0, 0.0, 0.0]

        train_metrics = summarize_metrics(*epoch_totals)
        validation_metrics = evaluate(model, validation_dl)
        epoch_history.append({
            "epoch": epoch + 1,
            "step": global_step,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        })
        print(
            f"validation loss={validation_metrics['loss']:.4f} "
            f"accuracy={validation_metrics['accuracy']:.4f} "
            f"perplexity={validation_metrics['perplexity']:.2f}"
        )
        save_checkpoint(model, checkpoint_dir / f"epoch_{epoch + 1:03d}")
        save_json(metrics_history, experiment_dir / "metrics.json")
        save_json(epoch_history, experiment_dir / "epoch_metrics.json")


    # Final checkpoint
    save_checkpoint(
        model,
        checkpoint_dir / "final"
    )

    print("\nTraining complete.")
    print(f"Results saved to {experiment_dir}")


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args(argv=None):
    # Config supplies defaults; explicit CLI arguments override them.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    known, _ = config_parser.parse_known_args(argv)
    defaults = {}
    if known.config:
        with known.config.open(encoding="utf-8") as f:
            defaults = json.load(f)

    parser = argparse.ArgumentParser(parents=[config_parser])

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name"
    )

    parser.add_argument(
        "--data",
        type=str,
        required="data" not in defaults
    )

    parser.add_argument(
        "--max-stories",
        type=int,
        default=10_000
    )

    parser.add_argument(
        "--maxlen",
        type=int,
        default=256
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=192
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=6
    )

    parser.add_argument(
        "--ff-dim",
        type=int,
        default=512
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=6
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--end-learning-rate",
        type=float,
        default=1e-5
    )

    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=50
    )

    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=123)

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument("--use-pre-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ffn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-final-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.set_defaults(**defaults)
    config = vars(parser.parse_args(argv))
    config.pop("config", None)
    return config


if __name__ == "__main__":
    config = parse_args()
    train(config)