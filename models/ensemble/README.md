# Ensemble Pipelines

These reference Triton ensemble models compose demodulation, speech recognition, and stream-processing components into end-to-end radio transcription flows. Both are decoupled and expose `INPUT` (`FP32`), optional `METADATA` (`STRING`), optional `RESET` and `FLUSH` (`BOOL`), and `OUTPUT` (`STRING`).

| Pipeline | Purpose | Maturity | Backend | Dependencies |
| --- | --- | --- | --- | --- |
| [`fm_radio_asr_pipeline`](fm_radio_asr_pipeline/) | Broadcast-FM radio-to-text pipeline. | Reference | Triton ensemble, decoupled | `fm_stereo_demod` → `voxtral-mini-stt` → `fm_document_batcher` |
| [`gmrs_asr_pipeline`](gmrs_asr_pipeline/) | GMRS radio-to-text pipeline. | Reference | Triton ensemble, decoupled | `fm_gmrs_demod` → `audio_overlap` → `voxtral-mini-stt` → `transcript_stitcher` → `fm_document_batcher` |

The dependencies listed above are Triton logical model names. For AirStack Edge, package, upload, and enable each dependency as its own content-level archive; the category layout is source organization only. See the [deployment guide](../../docs/deployment.md).
