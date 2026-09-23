# Deployment Guide

AirStack Edge is the supported deployment path for this repository. Follow the official
[AirStack Edge overview](https://docs.deepwave.ai/AirStack/Edge/overview/) and
[Triton inference application note](https://docs.deepwave.ai/AirStack/Edge/application_notes/triton-inference-server/)
for product-specific authentication, upload, model-loading, and client API instructions.

**Note:** While these models may run in a stand-alone NVIDIA Triton server, AirStack Edge is the
repository's only tested deployment path.

## Package and upload models

Models are deployed by creating a tarball archive of the model folder. AirStack Edge accepts one
model per archive. Supported archive formats are `tar`, `tgz`, `tbz2`, and `tzst`. Pack each archive
at the **content level**: `config.pbtxt`, numeric version directory, and all model payload files
must be at the archive root. Do not wrap those files in a parent directory named after the model.
Name the archive for its exact Triton logical model name (for example, `fm_gmrs_demod.tar.gz`). See
the official
[Uploading a Model](https://docs.deepwave.ai/AirStack/Edge/application_notes/triton-inference-server/#uploading-a-model)
section for the supported procedure.

For example, the archive made from `models/demodulation/fm_gmrs_demod/` must contain this at its top
level:

```text
config.pbtxt
0/
gmrs.py
```

### Example: package one dependency

From the repository root, package `fm_gmrs_demod` as a content-level archive:

```bash
tar -czvf fm_gmrs_demod.tar.gz -C models/demodulation/fm_gmrs_demod .
```

The archive contains `config.pbtxt`, the `0/` version directory, and `gmrs.py` at its top level. The
`fm_gmrs_demod.tar.gz` may then be uploaded to AirStack Edge for inference. Repeat this pattern for
every dependency and ensemble, using each model's own source directory and exact logical model name.

Some models may require specific deployment patterns or access to external files. Be sure to read the
README.md associated with each model for specific deployment requirements.

## Load and test

1. Choose a pipeline or individual model from [Model Categories](../models/README.md).
2. Identify all required logical dependencies and read each model README for its external-artifact
   requirements.
3. Package each required model as its own content-level archive, then upload and enable it using the
   official AirStack Edge instructions.
4. Confirm that every required model is `READY` before running an ensemble.
5. Follow the official
   [Basic UI Inference](https://docs.deepwave.ai/AirStack/Edge/tutorials/client_ui_inference/)
   workflow to open a compatible receive stream and register result delivery. For the decoupled
   radio ASR pipelines, use Edge `continuous` stream mode and configure at least one webhook.

## Pipeline dependencies

An ensemble archive contains its own configuration, while each logical dependency is uploaded and
enabled as its own model archive. The required dependency models will be listed in the README.md for
each ensemble model.
