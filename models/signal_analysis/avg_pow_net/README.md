# Average Power Network

Reference ONNX Runtime model that produces a single average-power-style feature from a variable-length `float32` input buffer. It is a standalone signal-analysis building block, not a checked-in ensemble dependency.

| Tensor | Direction | Type and shape | Required | Meaning |
| --- | --- | --- | --- | --- |
| `input_buffer` | Input | `FP32`, `[-1]` | Yes | Variable-length model input buffer. |
| `output_buffer` | Output | `FP32`, `[1]` | Yes | One output feature value. |

The configured backend is `onnxruntime_onnx` with `max_batch_size: 128`. The serialized `0/model.onnx` artifact is a Deepwave creation and is covered by the repository BSD-3-Clause license. Its SHA-256 is `9BC5643C8BBB2F588253197F9809AFDA226848B9EB4C2AE9BBCC8D1F5C62A479`.

The artifact's source and regeneration procedure are not yet documented. Treat this as a reference model and validate feature semantics, expected normalization, and suitability for a target signal workflow before use.
