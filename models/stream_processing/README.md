# Stream Processing Models

These reference Triton Python components hold request-stream state or reshape audio and text between processing stages. Stateful components must be kept with the same logical stream for correct reset, overlap, batching, or stitching behavior.

| Model | Purpose | Maturity | Backend | Pipeline dependency |
| --- | --- | --- | --- | --- |
| [`audio_batcher`](audio_batcher/) | Batch streaming audio into overlapping chunks. | Reference | Triton Python | Reusable building block; not used by the checked-in ensembles |
| [`audio_overlap`](audio_overlap/) | Prepend audio overlap across requests. | Reference | Triton Python | `gmrs_asr_pipeline` |
| [`fm_document_batcher`](fm_document_batcher/) | Batch and format transcript text and metadata into document output. | Reference | Triton Python, decoupled | `fm_radio_asr_pipeline`, `gmrs_asr_pipeline` |
| [`text_batcher`](text_batcher/) | Batch text and optional metadata. | Reference | Triton Python | Reusable building block; not used by the checked-in ensembles |
| [`transcript_stitcher`](transcript_stitcher/) | Remove overlap between successive transcript segments. | Reference | Triton Python | `gmrs_asr_pipeline` |

`audio_overlap` and `transcript_stitcher` accept an optional `RESET` input. `fm_document_batcher` accepts optional `FLUSH` and `RESET` inputs and is decoupled. `text_batcher` has no flush input: both configured character thresholds are `0`, so each non-empty request is emitted immediately.
