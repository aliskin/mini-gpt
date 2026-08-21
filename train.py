import argparse
import json
from datetime import datetime
from pathlib import Path
import tiktoken

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax

from model import MiniGPT
from story_dataset import load_and_preprocess_data


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

    token_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits,
        targets
    )

    # Shape: (1, seq_len)
    positions = jnp.arange(targets.shape[1])[None, :]

    # Only positions with a real next-token target contribute.
    #
    # If length = 5:
    #
    # input:   t0 t1 t2 t3 EOS PAD ...
    # target:  t1 t2 t3 EOS PAD PAD ...
    # mask:     1  1  1  1   0   0 ...
    #
    loss_mask = positions < (lengths[:, None] - 1)

    loss = (
        token_losses * loss_mask
    ).sum() / loss_mask.sum()

    predictions = jnp.argmax(logits, axis=-1)

    correct = (predictions == targets) * loss_mask

    accuracy = correct.sum() / loss_mask.sum()

    return loss, (logits, accuracy)


# ---------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------

@nnx.jit
def train_step(model, optimizer, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    (loss, (logits, accuracy)), grads = grad_fn(model, batch)

    optimizer.update(grads)

    return loss, accuracy


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
    tokenizer = tiktoken.get_encoding("gpt2")
    text_dl, batches_per_epoch = load_and_preprocess_data(
        file_path=config["data"],
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        maxlen=config["maxlen"],
        max_stories=config["max_stories"],
        num_epochs=config["epochs"],
        shuffle=True,
        seed=config["seed"],
    )


    # ----- model -----
    vocab_size = tokenizer.n_vocab
    model = MiniGPT(
        maxlen=config["maxlen"],
        vocab_size=vocab_size,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        feed_forward_dim=config["ff_dim"],
        num_transformer_blocks=config["num_blocks"],
        rngs=nnx.Rngs(config["seed"]),
    )


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

    global_step = 0

    for epoch in range(config["epochs"]):

        running_loss = 0.0
        running_steps = 0

        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        for batch in text_dl:

            batch = prepare_batch(batch)

            loss, accuracy = train_step(
                model,
                optimizer,
                batch
            )

            global_step += 1
            running_steps += 1
            running_loss += float(loss)

            if global_step % config["log_every"] == 0:

                mean_loss = running_loss / running_steps
                current_lr = float(lr_schedule(global_step))

                record = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": mean_loss,
                    "accuracy": float(accuracy),
                    "learning_rate": current_lr,
                }

                metrics_history.append(record)

                print(
                    f"step={global_step:5d} "
                    f"loss={mean_loss:.4f} "
                    f"accuracy={accuracy:.4f} "
                    f"lr={current_lr:.2e}"
                )

                running_loss = 0.0
                running_steps = 0


        # Save after every epoch
        save_checkpoint(
            model,
            checkpoint_dir / f"epoch_{epoch + 1:03d}"
        )

        save_json(
            metrics_history,
            experiment_dir / "metrics.json"
        )


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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    config = parse_args()
    train(config)