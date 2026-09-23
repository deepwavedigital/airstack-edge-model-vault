# Signal Analysis Models

These models are standalone signal-analysis building blocks rather than dependencies of the checked-in radio transcription ensembles.

| Model | Purpose | Maturity | Backend | Dependency information |
| --- | --- | --- | --- | --- |
| [`avg_pow_net`](avg_pow_net/) | Extract average-power features. Triton's logical model name is `avg_power_net`. | Reference | ONNX Runtime | No checked-in ensemble dependency |
| [`simple_fft`](simple_fft/) | Compute an FFT of interleaved IQ samples. | Experimental | LibTorch | No checked-in ensemble dependency |

`avg_pow_net` accepts `input_buffer` (`FP32`) and returns `output_buffer` (`FP32`). `simple_fft` accepts `INPUT__0` (`FP32`) with interleaved IQ samples and returns `OUTPUT__0` (`FP32`) shaped `[2, L]` for real and imaginary FFT values. `simple_fft` remains Experimental pending provenance and reproducibility documentation.
