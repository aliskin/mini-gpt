# Three architecture experiments

All runs use GPT-2 tokenization, the same 25K-story source, a 10% validation split
(split seed 123), model/training seed 42, context 256, batch size 32, embedding
width 192, 6 attention heads, 6 blocks, and 3 epochs. Learning rate, warmup,
weight decay, and logging settings also match.

| Experiment | Pre-LayerNorm | GELU FFN | Final LayerNorm | Comparison |
| --- | --- | --- | --- | --- |
| `attention_validation_25k_256` | Yes | No | No | Attention baseline |
| `ffn_validation_25k_256` | Yes | Yes | No | Versus attention: contribution of the FFN |
| `final_norm_validation_25k_256` | Yes | Yes | Yes | Versus FFN: contribution of final normalization |

The FFN comparison includes additional parameter capacity; it is not a
parameter-matched comparison. Final normalization adds only 384 parameters.
These are controlled comparisons at one seed; improvements are hypotheses.

## Run

Training now accepts `--config`. Explicit command-line arguments override values
in the file. Training writes the resolved configuration, including architecture
flags, tokenizer, and vocabulary size, back into the output experiment directory.

```sh
python train.py --config experiments/attention_validation_25k_256/config.json
python train.py --config experiments/ffn_validation_25k_256/config.json
python train.py --config experiments/final_norm_validation_25k_256/config.json
```

For a smoke check, retain the architecture flags from each config but override
the workload and output name. Repeat this command for each config, with distinct
smoke names:

```sh
python train.py --config experiments/attention_validation_25k_256/config.json \
  --name smoke_attention --max-stories 128 --epochs 2 --maxlen 32 \
  --batch-size 8 --embed-dim 32 --num-heads 2 --ff-dim 64 \
  --num-blocks 1 --log-every 2
```

For a shorter substantive comparison, override `--max-stories 5000 --maxlen 128`
on all three configs and give each run a distinct name reflecting those settings.
Update the notebook's `EXPERIMENTS` list accordingly. Do not mix those results
with the 25K/context-256 results.

## Evaluate

Run the imports and the "Compare the three architecture experiments" section in
`Evaluation.ipynb`. It plots train/validation loss, accuracy, and perplexity,
shows final-epoch metrics, and generates text using each model's own architecture.
Historical evaluation sections are optional.

Prioritize validation loss/perplexity at matching training steps. Accuracy
reports exact top-1 predictions and may move differently. Perplexity is `exp(loss)`
and adds interpretability rather than independent information. Growing gaps
between training and validation metrics suggest overfitting. Generated stories
can expose repetition and incoherence that scalar metrics miss.

Use the same prompts, temperature (0.8), and generation seed (42). Compare the
final epoch for all runs as the primary result; avoid choosing different epochs
only because one makes a variant look better.

`evaluation.load_experiment(name)` reconstructs models through the same
`model_from_config` function used by training. It also supports
`checkpoint="epoch_001"`. Older checkpoints without architecture flags are
recognized from their parameter metadata, including the original `norm` parameter
name. Explicit flags that contradict a checkpoint raise an error. Missing
checkpoints raise a clear error instead of evaluating random initialized weights.

```python
from evaluation import load_experiment
model, config, training_metrics = load_experiment("final_norm_validation_25k_256")
```
