# Model Categories

This directory organizes [NVIDIA Triton](https://github.com/triton-inference-server/server) model
source by function. Each model directory retains its Triton configuration and numeric version
payload. Its location here does not change its Triton logical model name.

| Category                                           | Purpose                                                      |
| -------------------------------------------------- | ------------------------------------------------------------ |
| [Demodulation](demodulation/README.md)             | Convert radio IQ data to audio and metadata.                 |
| [Speech recognition](speech_recognition/README.md) | Convert audio to text.                                       |
| [Signal analysis](signal_analysis/README.md)       | Provide power and spectrum-analysis building blocks.         |
| [Stream processing](stream_processing/README.md)   | Helper functions to manage state, batching, formatting, etc. |
| [Ensemble pipelines](ensemble/README.md)           | Multi-stage model examples.                                  |

These models are experimental and included for testing. They require additional review for production
environments.

The nested source layout is not an AirStack Edge upload package. Package each selected model as a
content-level archive and upload/enable ensemble dependencies individually; see the
[deployment guide](../docs/deployment.md). For example, to package the `fm_gmrs_demod` model to
upload to AirStack Edge:

```bash
tar -czvf fm_gmrs_demod.tar.gz -C models/demodulation/fm_gmrs_demod .
```
