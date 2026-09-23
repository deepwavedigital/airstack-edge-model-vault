# Model Authoring Guide

Use this guide when adding or substantially changing a model in this repository. It describes repository conventions but it does not replace Triton backend documentation or a release approval.

## Choose the backend

Use a standard Triton backend when the model artifact is directly supported. Select the Python backend for repository-owned runtime logic, such as processing that cannot be expressed by an existing standard backend. An ensemble is appropriate for a declarative tensor flow across logical models.
## Create the required model layout

At minimum, add a model directory with a Triton configuration and a numeric version directory:

```text
<model-directory>/
|-- README.md
|-- config.pbtxt
`-- <version>/
    `-- model.py or serialized model artifact
```

## Define and validate the interface

In `config.pbtxt`, declare every input and output with its exact name, data type, dimensions, and optionality. Also specify batching, instance settings, parameters, and transaction policy where relevant.

Before documenting an interface, validate it against both the configuration and the implementation or serialized artifact. Do not infer behavior from an older README. Record these details in the model README:

- Purpose, maturity, and backend;
- Triton logical model name;
- Inputs, outputs, and parameter defaults as deployed;
- Model dependencies and external artifacts; and
- Limitations, expected sample format, and known operational constraints.

If a desired change modifies tensor names, types, shapes, logical names, model parameters, response semantics, or ensemble mappings, treat it as a runtime interface change.
