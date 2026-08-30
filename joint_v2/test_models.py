import torch

from models import (
    ClassicalNeuralModel,
    HybridQuantumModel,
    count_trainable_parameters,
)


model = ClassicalNeuralModel()

example_features = torch.zeros(
    (5, 8),
    dtype=torch.float32,
)

with torch.no_grad():
    output = model(example_features)

print("Input shape:", example_features.shape)
print("Output shape:", output.shape)
print(
    "Trainable parameters:",
    count_trainable_parameters(model),
)
print(
    "All outputs are finite:",
    torch.isfinite(output).all().item(),
)

quantum_model = HybridQuantumModel()

with torch.no_grad():
    quantum_output = quantum_model(
        example_features
    )

print()
print(
    "Quantum output shape:",
    quantum_output.shape,
)
print(
    "Quantum trainable parameters:",
    count_trainable_parameters(
        quantum_model
    ),
)
print(
    "All quantum outputs are finite:",
    torch.isfinite(
        quantum_output
    ).all().item(),
)