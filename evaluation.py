"""Reconstruct experiment architectures and restore matching checkpoints."""
import json
from pathlib import Path

import flax.nnx as nnx
import orbax.checkpoint as ocp

from model import model_from_config


def load_experiment(name, *, experiments_dir="experiments", checkpoint="final"):
    """Return (model, resolved_config, training_metrics).

    New configs explicitly record architecture and vocabulary. For older runs,
    inspect checkpoint metadata to recover component presence and parameter names.
    Pass checkpoint='epoch_001' to inspect an intermediate trained model.
    """
    experiment_dir = Path(experiments_dir) / name
    config = json.loads((experiment_dir / "config.json").read_text())
    checkpoint_path = experiment_dir / "checkpoints" / checkpoint
    # The earliest notebook used a different checkpoint directory.
    if not checkpoint_path.exists() and checkpoint == "final":
        checkpoint_path = experiment_dir / "10000_checkpoint.orbax"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found for {name}: {checkpoint_path}")

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        metadata = checkpointer.metadata(checkpoint_path.resolve())
        tree = metadata.item_metadata.tree
        blocks = tree["transformer_blocks"]
        block = next(iter(blocks.values()))
        architecture = {
            "use_pre_norm": "norm1" in block or "norm" in block,
            "use_ffn": "ff1" in block,
            "use_final_norm": "final_norm" in tree,
            "attention_norm_name": "norm" if "norm" in block else "norm1",
        }
        for key, value in architecture.items():
            if key in config and config[key] != value:
                raise ValueError(f"Config/checkpoint mismatch for {key}: {config[key]} vs {value}")
            config[key] = value
        vocab_size = tree["embedding"]["token_emb"]["embedding"]["value"].shape[0]
        if "vocab_size" in config and config["vocab_size"] != vocab_size:
            raise ValueError("Config/checkpoint vocabulary sizes differ")
        config["vocab_size"] = vocab_size
        config.setdefault("tokenizer", "gpt2")
        model = model_from_config(config)
        state = checkpointer.restore(checkpoint_path.resolve(), item=nnx.state(model))
        nnx.update(model, state)
    finally:
        checkpointer.close()

    metrics_path = experiment_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    return model, config, metrics
