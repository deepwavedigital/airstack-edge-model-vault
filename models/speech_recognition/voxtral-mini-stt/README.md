# Voxtral Mini Speech to Text

Reference Triton Python wrapper for the INT4 Voxtral Mini 4B Realtime runtime. It converts one complete mono audio segment into transcript text for the checked-in radio transcription pipelines.

## Interface

| Tensor | Direction | Type and shape | Required | Meaning |
| --- | --- | --- | --- | --- |
| `AUDIO_IN` | Input | `FP32`, `[-1]` | Yes | Flattened mono audio samples. |
| `TEXT_OUT` | Output | `STRING`, `[1]` | Yes | Transcript text; empty input returns an empty string. |

The model is unbatched (`max_batch_size: 0`). It flattens input to `float32`, replaces NaN/Inf values, resamples from configured `input_sample_rate` to the runtime sample rate, and rejects audio over the configured duration limit.

## Deployed parameters

`device=cuda`, `max_tokens=256`, `compile=false`, `input_sample_rate=16000`, `max_audio_seconds=30.0`, `warmup=true`, `warmup_seconds=1.0`, `page_cache_policy=startup`, and `upstream_verbose=false`. The wrapper keeps upstream runtime diagnostics disabled by default; the implementation fallback is also `false` when the parameter is absent.

## Required artifacts

Two deployment files are omitted from Git and must be placed at these exact paths:

| File | Destination | Canonical download |
| --- | --- | --- |
| Model weights | `1/consolidated.safetensors` | [consolidated.safetensors](https://archive.deepwavedigital.com/triton-models/voxtral-mini-stt/consolidated.safetensors) |
| Packed environment | `voxtral-conda-env.tar.gz` | [voxtral-conda-env.tar.gz](https://archive.deepwavedigital.com/triton-models/voxtral-mini-stt/voxtral-conda-env.tar.gz) |

Download them from the repository root:

```bash
wget -P models/speech_recognition/voxtral-mini-stt/1/ https://archive.deepwavedigital.com/triton-models/voxtral-mini-stt/consolidated.safetensors
wget -P models/speech_recognition/voxtral-mini-stt/ https://archive.deepwavedigital.com/triton-models/voxtral-mini-stt/voxtral-conda-env.tar.gz
```

`EXECUTION_ENV_PATH` resolves the packed environment from the model root. The checked-in [environment specification](voxtral-conda.yml) and build instructions are retained below for maintainers; rebuilding the environment targets the historical Triton 24.08 Jetson container and CUDA architecture 8.7.

## Build a modified environment

The supplied environment is the default deployment path. A modified environment may require a build lasting more than one hour:

```bash
docker run --rm --platform linux/arm64 \
  --network=host \
  -v "$PWD:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/tritonserver:24.08-py3-igpu \
  bash -lc '
    apt-get update &&
    apt-get install -y --no-install-recommends bzip2 ca-certificates git wget &&
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh &&
    bash /tmp/miniforge.sh -b -p /opt/conda &&
    export PATH=/opt/conda/bin:$PATH &&
    mamba env create -f models/speech_recognition/voxtral-mini-stt/voxtral-conda.yml &&
    source /opt/conda/etc/profile.d/conda.sh &&
    conda activate voxtral-conda &&
    PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])") &&
    NVIDIA_LIBS=$(find "$PY_SITE/nvidia" -type d -name lib 2>/dev/null | paste -sd: -) &&
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}" &&
    export TORCH_CUDA_ARCH_LIST="8.7" &&
    python -m pip install --no-build-isolation "marlin @ git+https://github.com/IST-DASLab/marlin.git@1f25790bdd49fba53106164a24666dade68d7c90" &&
    find "$PY_SITE" -name "libcudss.so*" \( -type f -o -type l \) -print -exec cp -av {} "$CONDA_PREFIX/lib/" \; &&
    if [ ! -e "$CONDA_PREFIX/lib/libcudss.so.0" ]; then CUDSS_TARGET=$(find "$CONDA_PREFIX/lib" -maxdepth 1 -name "libcudss.so*" -type f | sort -V | tail -n 1); [ -n "$CUDSS_TARGET" ] || exit 1; ln -sf "$(basename "$CUDSS_TARGET")" "$CONDA_PREFIX/lib/libcudss.so.0"; fi &&
    conda-pack -n voxtral-conda -o /workspace/models/speech_recognition/voxtral-mini-stt/voxtral-conda-env.tar.gz
  '
```

Use the checked-in environment specification and review its dependencies before producing a replacement archive. This model is used by [FM Radio ASR Pipeline](../../ensemble/fm_radio_asr_pipeline/README.md) and [GMRS ASR Pipeline](../../ensemble/gmrs_asr_pipeline/README.md).
