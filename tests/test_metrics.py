import math
import unittest

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from model import MiniGPT
from story_dataset import create_dataloader, split_stories
from train import evaluate, loss_fn, summarize_metrics, train_step
import optax


class Tokenizer:
    def encode(self, text, **kwargs):
        if text == '<|endoftext|>':
            return [3]
        return [int(token) for token in text.split()]


class UniformModel(nnx.Module):
    def __call__(self, inputs):
        return jnp.zeros((*inputs.shape, 4))


class MetricsTest(unittest.TestCase):
    def test_mask_ignores_padding_and_missing_target(self):
        inputs = jnp.array([[0, 1, 2, 0], [0, 0, 0, 0]])
        targets = jnp.array([[1, 2, 0, 0], [0, 0, 0, 0]])
        loss, totals = loss_fn(UniformModel(), (inputs, targets, jnp.array([3, 2])))
        self.assertAlmostEqual(float(loss), math.log(4), places=6)
        self.assertEqual(int(totals[1]), 1)
        self.assertEqual(int(totals[2]), 3)
        self.assertAlmostEqual(summarize_metrics(*map(float, totals))['perplexity'], 4, places=5)

    def test_validation_retains_partial_batch_and_can_repeat(self):
        stories = ['0 1 2', '0 0', '0 1 2 3']
        loader = create_dataloader(stories, Tokenizer(), 2, 5, drop_remainder=False)
        for _ in range(2):
            metrics = evaluate(UniformModel(), loader)
            self.assertEqual(metrics['tokens'], 6)
            self.assertAlmostEqual(metrics['accuracy'], 1 / 6)
            self.assertAlmostEqual(metrics['perplexity'], 4, places=5)

    def test_split_reproducible_and_no_duplicate_leakage(self):
        stories = ['a', 'b', 'a', 'c', 'd', 'e']
        train, val = split_stories(stories, 0.4, 123)
        self.assertEqual((train, val), split_stories(stories, 0.4, 123))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train + val), set(stories))

    def test_training_and_evaluation_small_model(self):
        model = MiniGPT(5, 4, 8, 2, 16, 1, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adamw(1e-3))
        loader = create_dataloader(['0 1 2', '0 0', '0 1 2 3'], Tokenizer(), 2, 5,
                                   drop_remainder=False)
        from train import prepare_batch
        for batch in loader:
            totals = train_step(model, optimizer, prepare_batch(batch))
            self.assertTrue(all(np.isfinite(float(value)) for value in totals))
        before = [np.array(value.value) for _, value in nnx.to_flat_state(nnx.state(model, nnx.Param))]
        metrics = evaluate(model, loader)
        after = [np.array(value.value) for _, value in nnx.to_flat_state(nnx.state(model, nnx.Param))]
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(metrics['tokens'], 6)
        self.assertTrue(math.isfinite(metrics['perplexity']))


if __name__ == '__main__':
    unittest.main()
