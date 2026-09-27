# Hybrid quantum classifier knowledge

## Beginner explanation

A qubit is the basic information unit used in a quantum circuit. In this
project, the qubits are mathematical simulations on a classical computer. The
eight input numbers are encoded as rotation angles, the circuit applies
trainable rotations and entangling gates, and measurements are passed to a
small classical output layer.

The model is called hybrid because quantum-style circuit computation and
classical neural-network computation are used together.

## Exact project circuit

The downstream input is transformed with `2 * arctan(feature)`. The model uses
four qubits and two data-reuploading rounds:

1. Encode features 1-4 with Y rotations on four qubits.
2. Apply one StronglyEntanglingLayers block.
3. Encode features 5-8 with Y rotations.
4. Apply a second StronglyEntanglingLayers block.
5. Measure Pauli-Z expectation values on all four qubits.
6. Feed the four measurements into a classical linear 4-to-1 output.

The circuit runs on PennyLane `default.qubit` and integrates with PyTorch
automatic differentiation. It has 29 trainable parameters.

## Data re-uploading

Data re-uploading alternates input-dependent gates with trainable circuit
operations. Repeated encoding can increase the range of functions represented
by a small circuit. The project uses two uploads because eight features are
encoded four at a time on four qubits.

## Entangling layers

PennyLane StronglyEntanglingLayers combine parameterised single-qubit rotations
with two-qubit entanglers, normally CNOT gates. Entanglement allows the circuit
to represent interactions between qubits/features. Its presence does not by
itself prove useful quantum advantage.

## Simulation versus quantum hardware

`default.qubit` is a software simulator. Therefore:

- The reported results demonstrate behaviour of a variational quantum-circuit
  model, not a physical quantum processor.
- Hardware noise and device constraints were not evaluated.
- The simulation does not establish computational speedup.
- Strong predictive results do not establish quantum advantage.

## Why compare against 31 classical parameters?

The classical comparator has 31 parameters and the hybrid model has 29. Close
parameter counts reduce the risk that one model wins simply because it is much
larger. Parameter matching does not make every aspect of model capacity or
computational cost identical, so the comparison should be called controlled,
not perfectly equivalent.

## Project conclusion

The hybrid model learned predictive signal and remained close to the classical
model, but it had lower mean locked-test ROC-AUC on both tasks. The correct
statement is that no quantum advantage was demonstrated in this experiment.
It is not valid to conclude that all quantum models are worse, because only one
small architecture, one transformation, and one feature representation were
tested.

## Primary sources

- Project model implementation: joint_v2/models.py
- Project documentation: joint_v2/README.md
- PennyLane StronglyEntanglingLayers: https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html
- PennyLane PyTorch interface: https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html
- Data-reuploading paper: https://arxiv.org/abs/1907.02085

