# Triton Model Repository Concepts

This repository is a source collection for
[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) models to run in
AirStack Edge. For AirStack Edge context, see the
[AirStack Edge overview](https://docs.deepwave.ai/AirStack/Edge/overview/) and its
[Triton inference application note](https://docs.deepwave.ai/AirStack/Edge/application_notes/triton-inference-server/).

## Model repositories and versions

AirStack Edge uses a Triton Inference Server to execute AI and signal processing pipelines. Triton loads a model from a directory whose direct child name is the model's logical name. That
directory contains a `config.pbtxt` and one or more numeric version directories structured as follows:

```text
 logical-model-name/
 |-- config.pbtxt
 `-- 0/
     `-- model artifact or implementation
```

The `name` in `config.pbtxt` is the logical model name used by clients and ensemble dependencies. A
version directory (`0`, `1`, and so on) holds one version of that model. Version numbers are Triton
deployment versions, not this repository's release version.

## Configurations define the deployed interface

`config.pbtxt` declares the deployed interface and runtime behavior, including:

- Logical model name and backend or platform
- Input and output tensor names, data types, shapes, and optionality
- Batching and instance settings
- Model parameters
- Ensemble scheduling or decoupled transaction settings, when applicable

## Backends

An
[NVIDIA Triton Backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/backend/docs/backend_platform_support_matrix.html)
is the runtime that executes a model. This repository includes these patterns:

| Pattern        | Configuration field            | Example                 | Typical payload                                                |
| -------------- | ------------------------------ | ----------------------- | -------------------------------------------------------------- |
| Python backend | `backend: "python"`            | `voxtral-mini-stt`      | `model.py` and any declared execution environment or artifacts |
| ONNX Runtime   | `platform: "onnxruntime_onnx"` | `avg_power_net`         | an ONNX model file                                             |
| LibTorch       | `platform: "pytorch_libtorch"` | `simple_fft`            | a TorchScript model file                                       |
| Ensemble       | `platform: "ensemble"`         | `fm_radio_asr_pipeline` | configuration that maps tensors between other logical models   |

## Ensemble Models as Processing Pipelines

An ensemble has no standalone inference artifact. Its `ensemble_scheduling` section connects named
tensors between logical models. For AirStack Edge, upload and enable every dependency named by
`model_name` as its own model archive.

For example, `fm_radio_asr_pipeline` connects `fm_stereo_demod`, `voxtral-mini-stt`, and
`fm_document_batcher`. Moving source directories within `models/` must never change those logical
names or the names used in an ensemble configuration.

Some pipelines are decoupled: a request can produce zero, one, or multiple responses over time. The
included radio transcription ensembles are decoupled, so AirStack Edge must run them in `continuous`
stream mode and send results to at least one configured webhook. In this workflow Edge supplies
`INPUT` and optional `METADATA` and it ignores the optional `RESET` and `FLUSH` ensemble inputs.

## Before you deploy

1. Choose a model or pipeline from the [model categories](../models/README.md).
2. Read its configuration and model documentation to identify its inputs, outputs, parameters,
   dependencies, maturity, and required artifacts.
3. Package each selected model separately, preserving its numeric version directory and payload.
4. Obtain external artifacts only from their documented source and satisfy their separate license
   terms.
5. Use the model's documented client interface, particularly for stateful or decoupled streaming
   models.

This guide explains the concepts. For the supported procedure, use the repository
[deployment guide](deployment.md) together with the official AirStack Edge documentation.
Compatibility matrices and artifact checksums are outside this guide.
