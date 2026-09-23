# Contributing

Thank you for helping improve AirStack Edge Model Vault. Contributions should preserve the repository's role as a collection of reference and experimental Triton model components for the supported AirStack Edge deployment path.

## Before you start

- Open an issue first for a proposed model, interface change, or substantial behavior change.
- Keep a model in one `models/<category>/<model-directory>/` home. Its `config.pbtxt` logical model name is a deployment contract; do not rename it or change tensor names, types, shapes, parameters, response behavior, or ensemble mappings without explicit approval.
- Follow the [model authoring guide](docs/model-authoring.md) and [deployment guide](docs/deployment.md). Do not commit downloaded weights, packed environments, credentials, or unreviewed third-party artifacts.

## Pull requests

1. Update the affected local README and category index.
2. Run validation appropriate to the change.
3. Explain the user-visible effect, validation performed, artifact/provenance impact, and any AirStack Edge packaging impact in the pull request.

By submitting a contribution, you confirm that you have the right to submit it and agree to license it under the repository's BSD-3-Clause terms, unless the contribution contains clearly identified third-party material accepted by the maintainers under its own compatible license.
