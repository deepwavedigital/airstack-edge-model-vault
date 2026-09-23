# Audio Overlap

Reference Triton Python stage that prepends prior audio to each new segment so downstream transcription can span segment boundaries. It is used by [GMRS ASR Pipeline](../../ensemble/gmrs_asr_pipeline/README.md).

| Tensor | Direction | Type and shape | Required | Meaning |
| --- | --- | --- | --- | --- |
| `AUDIO_IN` | Input | `FP32`, `[-1]` | Yes | New audio segment. |
| `RESET` | Input | `BOOL`, `[1]` | No | Discard the prior tail before processing. |
| `AUDIO_OUT` | Output | `FP32`, `[-1]` | Yes | Prior tail followed by the new audio. |

The model is unbatched and stateful by correlation ID. With the deployed 16 kHz sample rate and two-second overlap, it retains the last 32,000 samples of each incoming segment. `RESET=true` clears the prior tail, then the current audio becomes the next tail.

Output duplicates samples across adjacent requests by design. Pair it with overlap-aware downstream processing such as `transcript_stitcher`; do not treat its output as a concatenation-safe archival audio stream.
