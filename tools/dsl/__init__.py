"""DSL primitive library for ARC-AGI style grid transformations.

Every primitive is a `(np.ndarray[H,W] int) -> np.ndarray[H',W'] int` Python
reference function paired with an ONNX subgraph that operates on the
competition's `[1, 10, 30, 30]` one-hot BOOL/FLOAT representation.

See `primitives.py` for primitive builders, `compiler.py` for program
compilation.
"""
