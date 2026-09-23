# Simple FFT

Experimental LibTorch model that calculates an FFT-shifted complex spectrum from interleaved IQ samples. It is a standalone signal-analysis building block, not a checked-in ensemble dependency.

| Tensor | Direction | Type and shape | Required | Meaning |
| --- | --- | --- | --- | --- |
| `INPUT__0` | Input | `FP32`, `[-1]` | Yes | Interleaved IQ samples (`[2L]`). |
| `OUTPUT__0` | Output | `FP32`, `[2, -1]` | Yes | FFT-shifted spectrum with real and imaginary rows (`[2, L]`). |

The configured backend is `pytorch_libtorch` and the model is unbatched. The checked-in TorchScript artifact's `forward(x)` converts interleaved samples to complex values, applies FFT shift and FFT, then stacks real and imaginary outputs. It does not accept SigMF metadata or return JSON.

`0/model.pt` is a Deepwave creation covered by the repository BSD-3-Clause license. Its SHA-256 is `7A6FBDC0044CB589F16DC78A47B0AC6BC679EE27AB56E229F328945801C4A6B4`. The artifact source and regeneration procedure remain unknown, so retain its Experimental status and validate it before use.
