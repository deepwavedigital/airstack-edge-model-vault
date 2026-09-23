# Speech Recognition Models

This category contains the speech-recognition stage used by the radio transcription pipelines.

| Model | Purpose | Maturity | Backend | Used by |
| --- | --- | --- | --- | --- |
| [`voxtral-mini-stt`](voxtral-mini-stt/) | Transcribe 16 kHz audio to text with the Voxtral Mini runtime. | Reference | Triton Python | `fm_radio_asr_pipeline`, `gmrs_asr_pipeline` |

The configured model accepts `AUDIO_IN` (`FP32`) and returns `TEXT_OUT` (`STRING`). Its deployment requires externally acquired weights and a packed execution environment; use the model-specific README for the current acquisition and build instructions. The checked-in configuration enables upstream diagnostic verbosity; this remains a separately approved runtime-configuration concern.
