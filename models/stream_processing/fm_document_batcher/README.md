# FM Document Batcher

Reference, decoupled Triton Python model that accumulates transcript text and emits compact document JSON with selected radio metadata. It is the final stage of both checked-in radio ASR ensembles.

## Interface

| Tensor | Direction | Type and shape | Required | Meaning |
| --- | --- | --- | --- | --- |
| `TEXT_IN` | Input | `STRING`, `[1]` | Yes | Transcript segment to add to the stream document. |
| `METADATA_IN` | Input | `STRING`, `[1]` | No | SigMF metadata or enriched document metadata JSON. |
| `FLUSH` | Input | `BOOL`, `[1]` | No | Emit buffered non-empty text immediately. |
| `RESET` | Input | `BOOL`, `[1]` | No | Discard retained state before processing the request. |
| `OUTPUT` | Output | `STRING`, `[1]` | Conditional | Compact JSON document when emission conditions are met. |

The model is unbatched and decoupled. Within the supported AirStack Edge path it runs as an ensemble stage; Edge continuous-mode results are delivered to at least one configured webhook. Edge currently does not supply this model's optional `RESET` or `FLUSH` controls through the outer ensemble.

## Batching and output

The deployed parameters are `min_chars_per_batch=400`, `max_chars_per_batch=2000`, `stream_prefix=fm`, and an empty configured `stream_id`. In the model interface, a non-empty document emits at the minimum threshold, maximum threshold, or `FLUSH=true`; `RESET=true` clears state before incoming text is added. The supported Edge path does not currently send either optional control, so only the thresholds trigger emission there.

The response object contains `document`, monotonically increasing `doc_index`, and `doc_metadata`. Metadata includes `streamId`, start/end NTP timestamps, and sanitized `radio_metadata`. SigMF declaration fields and empty containers are removed. Input enriched metadata can provide `streamId` and timestamp fields.

Used by [FM Radio ASR Pipeline](../../ensemble/fm_radio_asr_pipeline/README.md) and [GMRS ASR Pipeline](../../ensemble/gmrs_asr_pipeline/README.md). It must not be treated as a stateless request/response formatter.
