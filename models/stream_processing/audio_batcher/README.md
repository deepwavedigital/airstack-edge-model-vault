# Audio Batcher

Reference Triton Python utility for accumulating audio chunks into overlapping windows. It is not used by the checked-in ensembles.

| Tensor | Direction | Type and shape | Required | Meaning |
| --- | --- | --- | --- | --- |
| `AUDIO_IN` | Input | `FP32`, `[-1]` | Yes | Audio samples to append to the correlation-ID stream buffer. |
| `FLUSH` | Input | `BOOL`, `[1]` | No | Emit all buffered audio immediately. |
| `AUDIO_OUT` | Output | `FP32`, `[-1]` | Yes | A full window, flushed remainder, or empty tensor. |

The model is unbatched and stateful by Triton correlation ID. Its deployed parameters are a 16 kHz sample rate, 10-second chunk, 2-second overlap, and 60-second maximum retained buffer. A full chunk advances the buffer by eight seconds; `FLUSH=true` emits the buffered remainder and clears it. Without either condition, it returns an empty `AUDIO_OUT` tensor.

Keep each logical stream on a stable correlation ID. There is no `RESET` input; a new correlation ID is the only documented way to avoid using retained buffer state.
