import pennylane as qml
import torch
from torch import nn


class ClassicalNeuralModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 3),
            nn.Tanh(),
            nn.Linear(3, 1),
        )

    def forward(self, features):
        logits = self.network(features)

        return logits.squeeze(-1)


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

NUMBER_OF_QUBITS = 4
NUMBER_OF_QUANTUM_LAYERS = 2

quantum_device = qml.device(
    "default.qubit",
    wires=NUMBER_OF_QUBITS,
)


@qml.qnode(
    quantum_device,
    interface="torch",
    diff_method="backprop",
)
def quantum_circuit(inputs, weights):
    for wire in range(NUMBER_OF_QUBITS):
        qml.RY(
            inputs[..., wire],
            wires=wire,
        )

    qml.StronglyEntanglingLayers(
        weights[0:1],
        wires=range(NUMBER_OF_QUBITS),
    )

    for wire in range(NUMBER_OF_QUBITS):
        qml.RY(
            inputs[..., wire + 4],
            wires=wire,
        )

    qml.StronglyEntanglingLayers(
        weights[1:2],
        wires=range(NUMBER_OF_QUBITS),
    )

    return [
        qml.expval(
            qml.PauliZ(wire)
        )
        for wire in range(NUMBER_OF_QUBITS)
    ]

class HybridQuantumModel(nn.Module):
    def __init__(self):
        super().__init__()

        weight_shapes = {
            "weights": (
                NUMBER_OF_QUANTUM_LAYERS,
                NUMBER_OF_QUBITS,
                3,
            )
        }

        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes,
        )

        self.output_layer = nn.Linear(
            NUMBER_OF_QUBITS,
            1,
        )

    def forward(self, features):
        quantum_outputs = self.quantum_layer(
            features
        )

        quantum_outputs = quantum_outputs.to(
            dtype=torch.float32
        )

        logits = self.output_layer(
            quantum_outputs
        )

        return logits.squeeze(-1)