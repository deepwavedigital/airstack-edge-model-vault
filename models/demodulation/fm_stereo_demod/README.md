# FM Stereo Demodulator

Reference Triton Python model that demodulates broadcast-FM interleaved IQ into mono audio for the FM radio ASR pipeline.

## Interface

| Tensor | Direction | Type and shape | Required | Meaning |
| --- | --- | --- | --- | --- |
| `IQ_IN` | Input | `FP32`, `[-1]` | Yes | Interleaved real/imaginary IQ values. |
| `META_IN` | Input | `STRING`, `[1]` | No | Metadata passed through unchanged. |
| `AUDIO_OUT` | Output | `FP32`, `[-1]` | Yes | One-dimensional mono audio. |
| `META_OUT` | Output | `STRING`, `[1]` | Yes | `META_IN`, or an empty string when omitted. |

The deployed configuration is unbatched and runs a CPU Python backend instance. It sets a 1,920,000 Hz IQ rate, 16,000 Hz audio rate, and 192,000 Hz `demod_rate` parameter.

## Current behavior and limitations

The internal GNU Radio flowgraph recovers left and right channels, but `process()` sums them and returns mono `AUDIO_OUT`. The configured `demod_rate` controls selection of the integer decimation nearest the requested FM intermediate rate. Its deployed 192,000 Hz value preserves the previous default decimation of 10 and intermediate rate of 192,000 Hz.

The model passes metadata through without logging it and does not write host-local debug audio.

Used by [FM Radio ASR Pipeline](../../ensemble/fm_radio_asr_pipeline/README.md).
