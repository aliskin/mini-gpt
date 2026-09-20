import json
from pathlib import Path
import tempfile
import unittest

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from evaluation import load_experiment
from model import model_from_config
from train import parse_args


class ExperimentTest(unittest.TestCase):
    def test_config_cli_overrides(self):
        path = 'experiments/attention_validation_25k_256/config.json'
        config = parse_args(['--config', path, '--epochs', '2'])
        self.assertFalse(config['use_ffn'])
        self.assertFalse(config['use_final_norm'])
        self.assertEqual(config['epochs'], 2)
        self.assertEqual(config['max_stories'], 25000)
        self.assertNotIn('config', config)

    def test_checkpoint_round_trips_and_legacy_inference(self):
        variants = [(True, False, False, 'norm1'), (True, True, False, 'norm1'),
                    (True, True, True, 'norm1'), (False, False, False, 'norm1'),
                    (True, False, False, 'norm')]
        with tempfile.TemporaryDirectory() as directory:
            for i, (pre, ffn, final, norm_name) in enumerate(variants):
                with self.subTest(variant=i):
                    config = dict(maxlen=4, vocab_size=7, embed_dim=8, num_heads=2,
                                  ff_dim=16, num_blocks=2, seed=42,
                                  use_pre_norm=pre, use_ffn=ffn, use_final_norm=final,
                                  attention_norm_name=norm_name)
                    original = model_from_config(config)
                    inputs = jnp.array([[1, 2, 3, 0]])
                    expected = np.asarray(original(inputs))
                    experiment = Path(directory) / str(i)
                    experiment.mkdir()
                    checkpointer = ocp.PyTreeCheckpointer()
                    try:
                        checkpointer.save((experiment / 'checkpoints' / 'final').resolve(), nnx.state(original))
                    finally:
                        checkpointer.close()
                    # Test explicitly configured runs, then legacy metadata recovery.
                    for legacy in (False, True):
                        stored = dict(config)
                        if legacy:
                            for key in ('use_pre_norm', 'use_ffn', 'use_final_norm',
                                        'attention_norm_name', 'vocab_size'):
                                stored.pop(key)
                        (experiment / 'config.json').write_text(json.dumps(stored))
                        restored, resolved, metrics = load_experiment(str(i), experiments_dir=directory)
                        np.testing.assert_array_equal(expected, np.asarray(restored(inputs)))
                        self.assertEqual(resolved['use_ffn'], ffn)
                        self.assertEqual(resolved['use_final_norm'], final)
                        self.assertEqual(metrics, [])
                    stored['use_final_norm'] = not final
                    (experiment / 'config.json').write_text(json.dumps(stored))
                    with self.assertRaisesRegex(ValueError, 'mismatch'):
                        load_experiment(str(i), experiments_dir=directory)


if __name__ == '__main__':
    unittest.main()
