"""Differentiable dense-angle circuit using PyTorch on CPU or CUDA.

Wire zero is the most significant bit. The rotations and ring of CNOTs
match PennyLane's StronglyEntanglingLayers default range for each layer.
"""
import torch
from torch import nn


def _rotate(state, wire, qubits, matrix):
    # Move the selected two-level axis last and apply the same 2x2 gate
    # to every pair of amplitudes across the whole batch.
    shaped = state.reshape(state.shape[0], *([2] * qubits))
    pairs = shaped.movedim(wire + 1, -1)
    return (pairs @ matrix.T).movedim(-1, wire + 1).reshape_as(state)


class DenseStatevector(nn.Module):
    def __init__(self, qubits, layers):
        super().__init__()
        if not 2 <= qubits <= 20 or layers < 1:
            raise ValueError("Expected 2–20 wires and at least one entangling layer")
        self.qubits = qubits
        self.input_dim = 2 * qubits
        self.weights = nn.Parameter(torch.empty(layers, qubits, 3))
        nn.init.uniform_(self.weights, -0.3, 0.3)
        for layer in range(layers):
            # Compose the inverse CNOT permutation once, outside training.
            indices = torch.arange(1 << qubits)
            distance = 1 + layer % (qubits - 1)
            for control in reversed(range(qubits)):
                target = (control + distance) % qubits
                bit = (indices >> (qubits - 1 - control)) & 1
                indices = indices ^ (bit << (qubits - 1 - target))
            self.register_buffer(f"permutation_{layer}", indices, persistent=False)

    def forward(self, features):
        angles = (torch.pi * torch.tanh(features)).reshape(-1, self.qubits, 2)
        theta, phi = angles.unbind(-1)
        # RZ(phi) RY(theta) |0>, including both phase factors.
        zero = torch.cos(theta / 2) * torch.exp(-0.5j * phi)
        one = torch.sin(theta / 2) * torch.exp(0.5j * phi)
        local = torch.stack((zero, one), dim=-1)
        state = local[:, 0]
        for wire in range(1, self.qubits):
            state = (state.unsqueeze(-1) * local[:, wire, None, :]).flatten(1)
        for layer, params in enumerate(self.weights):
            for wire, (alpha, beta, gamma) in enumerate(params.unbind()):
                # Rot = RZ(gamma) RY(beta) RZ(alpha).
                c, s = torch.cos(beta / 2), torch.sin(beta / 2)
                a = c * torch.exp(-0.5j * (alpha + gamma))
                b = -s * torch.exp(0.5j * (alpha - gamma))
                matrix = torch.stack((a, b, -b.conj(), a.conj())).reshape(2, 2)
                state = _rotate(state, wire, self.qubits, matrix)
            state = state.index_select(1, getattr(self, f"permutation_{layer}"))
        probabilities = state.abs().square()
        readings = []
        for wire in range(self.qubits):
            marginal = probabilities.reshape(len(state), 2 ** wire, 2, -1).sum((1, 3))
            readings.append(marginal[:, 0] - marginal[:, 1])
        return torch.stack(readings, dim=1)
