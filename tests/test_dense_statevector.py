"""Numerical and training checks for the opt-in dense-angle branch."""
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from qmlpipe.dense_statevector import DenseStatevector
from qmlpipe.fresh_hybrid import build_hybrid_model, train_hybrid_from_features
from scripts import run_dense_hybrid as runner


def reference_readings(inputs, weights):
    """Independent small-system reference using full Kronecker matrices."""
    def ry(angle):
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        return np.array([[c, -s], [s, c]])

    def rz(angle):
        return np.diag(np.exp(np.array([-0.5j, 0.5j]) * angle))

    qubits = weights.shape[1]
    basis = np.arange(2 ** qubits)
    wire_bits = [1 << (qubits - 1 - wire) for wire in range(qubits)]
    state = np.array([1.0])
    for theta, phi in (np.pi * np.tanh(inputs)).reshape(qubits, 2):
        state = np.kron(state, (rz(phi) @ ry(theta))[:, 0])
    for layer, parameters in enumerate(weights):
        block = np.array([[1.0]])
        for alpha, beta, gamma in parameters:
            block = np.kron(block, rz(gamma) @ ry(beta) @ rz(alpha))
        state = block @ state
        for wire, control in enumerate(wire_bits):
            target = wire_bits[(wire + 1 + layer % (qubits - 1)) % qubits]
            activated = (basis & control) != 0
            after = state.copy()
            after[basis[activated] ^ target] = state[activated]
            state = after
    probabilities = np.abs(state) ** 2
    return np.array([probabilities @ np.where((basis & bit) == 0, 1, -1) for bit in wire_bits])


class DenseTests(unittest.TestCase):
    def test_gates_and_derivatives(self):
        torch.manual_seed(11)
        circuit = DenseStatevector(5, 2).double()
        x = torch.randn(2, 10, dtype=torch.float64, requires_grad=True)
        results = circuit(x)
        weights = circuit.weights.detach().numpy().copy()
        rows = x.detach().numpy().copy()
        expected = np.stack([reference_readings(row, weights) for row in rows])
        np.testing.assert_allclose(results.detach(), expected, atol=1e-11)
        results[0].sum().backward()
        eps = 1e-5
        for index in (0, 1):
            plus, minus = rows[0].copy(), rows[0].copy()
            plus[index] += eps
            minus[index] -= eps
            grad = (reference_readings(plus, weights) - reference_readings(minus, weights)).sum() / (2 * eps)
            self.assertAlmostEqual(x.grad[0, index].item(), grad, places=7)
        idx = (0, 1, 1)
        plus, minus = weights.copy(), weights.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad = (reference_readings(rows[0], plus) - reference_readings(rows[0], minus)).sum() / (2 * eps)
        self.assertAlmostEqual(circuit.weights.grad[idx].item(), grad, places=7)

    def test_scaling_and_matched_control(self):
        cfg = dict(embedding="dense_angle", qubits=5, layers=2, epochs=2,
                   validation_patience=2, max_batches_per_epoch=0, batch_size=4)
        rng = np.random.default_rng(0)
        train = rng.normal(size=(12, 8)).astype("float32")
        train[:, 0] = 7
        val = (rng.normal(size=(4, 8)) + 2).astype("float32")
        labels = np.arange(len(train)) % 2
        model, history = train_hybrid_from_features(train, labels, val, [0, 1, 0, 1], cfg, seed=1)
        np.testing.assert_allclose(model.feature_mean, train.mean(0), atol=1e-6)
        self.assertEqual(float(model.feature_scale[0]), 1)
        self.assertEqual([v["n_batches"] for v in history["epochs"]], [3, 3])
        selected = min(history["epochs"], key=lambda v: v["val_loss"])
        self.assertEqual(history["selected_epoch"], selected["epoch"])
        with torch.no_grad():
            actual = torch.nn.functional.cross_entropy(model(torch.from_numpy(val)), torch.tensor([0, 1, 0, 1]))
        self.assertAlmostEqual(float(actual), selected["val_loss"], places=6)
        other = build_hybrid_model(dict(cfg, arm="replacement"), 8)
        self.assertEqual(model.trainable_parameter_counts(), other.trainable_parameter_counts())
        only_classical = build_hybrid_model(dict(cfg, arm="classical"), 8)
        self.assertEqual(only_classical.trainable_parameter_counts()["quantum"], 0)

    def test_final_loads_validation_checkpoint_and_rejects_changed_rows(self):
        rng = np.random.default_rng(6)
        sizes = {"train": 16, "val": 6, "test": 8}
        data = {
            "X": {k: rng.normal(size=(n, 8)).astype("float32") for k, n in sizes.items()},
            "y": {k: np.arange(n) % 2 for k, n in sizes.items()},
            "sample_ids": {k: [f"{k}-{i}" for i in range(n)] for k, n in sizes.items()},
        }
        def invoke(*args):
            with patch.object(sys, "argv", ["run_dense_hybrid", *map(str, args)]), redirect_stdout(StringIO()):
                runner.main()

        estimator = Mock()
        estimator.predict.side_effect = lambda x: np.zeros(len(x), dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp, patch.object(runner.fp, "cache_features", return_value=data), \
                patch.object(runner.fp, "_make_estimator", return_value=estimator):
            val, final = Path(temp) / "val", Path(temp) / "final"
            invoke("validate", "--epochs", 1, "--output", val)
            self.assertEqual(json.loads((val / "metrics.json").read_text())["split"], "val")
            with patch.object(runner.hybrid, "train_hybrid_from_features", side_effect=AssertionError("refit")):
                invoke("final", "--checkpoint", val / "best.pt", "--output", final)
            report = json.loads((final / "metrics.json").read_text())
            self.assertEqual(report["split"], "test")
            with np.load(final / "paired_predictions.npz") as pred:
                np.testing.assert_array_equal(pred["sample_ids"], data["sample_ids"]["test"])
                self.assertEqual(report["accuracy"], (pred["model"] == pred["labels"]).mean())
            data["sample_ids"]["test"].reverse()
            with self.assertRaisesRegex(ValueError, "sample IDs"):
                invoke("final", "--checkpoint", val / "best.pt", "--output", Path(temp) / "changed")


if __name__ == "__main__":
    unittest.main()
