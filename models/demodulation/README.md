# Demodulation Models

These reference Triton Python models convert interleaved radio IQ samples to mono audio while carrying optional metadata through the processing stage. They are used by the ensemble pipelines.

| Model | Purpose | Maturity | Backend | Used by |
| --- | --- | --- | --- | --- |
| [`fm_gmrs_demod`](fm_gmrs_demod/) | Narrowband FM/GMRS demodulation. | Reference | Triton Python | `gmrs_asr_pipeline` |
| [`fm_stereo_demod`](fm_stereo_demod/) | Broadcast-FM demodulation; the current deployed output is mono audio. | Reference | Triton Python | `fm_radio_asr_pipeline` |

Both models accept `IQ_IN` (`FP32`) and optional `META_IN` (`STRING`), and produce `AUDIO_OUT` (`FP32`) plus `META_OUT` (`STRING`). Their checked-in configurations define a 1.92 MHz input sample rate and a 16 kHz audio rate.

See [ensemble pipelines](../ensemble/README.md) for the complete dependency chains.
