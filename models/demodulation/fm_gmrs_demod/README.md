# GMRS FM Demodulator

Reference Triton Python model that demodulates narrowband FM/GMRS interleaved IQ into mono audio for the GMRS ASR pipeline.

## Interface

| Tensor | Direction | Type and shape | Required | Meaning |
| --- | --- | --- | --- | --- |
| `IQ_IN` | Input | `FP32`, `[-1]` | Yes | Interleaved real/imaginary IQ values; length must be even. |
| `META_IN` | Input | `STRING`, `[1]` | No | Metadata to pass through unchanged. |
| `AUDIO_OUT` | Output | `FP32`, `[-1]` | Yes | Demodulated mono audio. |
| `META_OUT` | Output | `STRING`, `[1]` | Yes | `META_IN`, or `{}` when omitted. |

The model is unbatched and uses a CPU Python backend instance. The deployed parameters are 1,920,000 Hz input sampling, 16,000 Hz audio, decimation values 20 and 6, 6 kHz low-pass cutoff, 2 kHz transition, -75 dB squelch, 5 kHz maximum deviation, and 750 µs de-emphasis.

## Use and limitations

Pass contiguous interleaved `float32` IQ. Invalid odd-length IQ produces a Triton error. The model retains a GNU Radio flowgraph while loaded but does not expose client-controlled stream state. `META_OUT` only carries input metadata; it does not infer or validate radio metadata.

Used by [GMRS ASR Pipeline](../../ensemble/gmrs_asr_pipeline/README.md). The included `gmrs.py` is GNU Radio-related GPL-3.0 material; see repository licensing before redistributing or modifying the model.
