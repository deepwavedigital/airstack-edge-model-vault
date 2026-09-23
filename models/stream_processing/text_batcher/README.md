# Text Batcher

Reference Triton Python utility that turns text and optional metadata into a JSON `documents` array. It is not used by the checked-in ensembles.

| Tensor | Direction | Type and shape | Required | Meaning |
| --- | --- | --- | --- | --- |
| `TEXT_IN` | Input | `STRING`, `[1]` | Yes | Text to add to the correlation-ID stream buffer. |
| `METADATA_IN` | Input | `STRING`, `[1]` | No | JSON metadata merged into the buffered document state. |
| `OUTPUT` | Output | `STRING`, `[1]` | Yes | `{"documents":[...]}` JSON; the array can be empty. |

The model is unbatched and stateful by correlation ID. Its deployed `min_chars_per_batch` and `max_chars_per_batch` are both `0`, so each non-empty request emits immediately and clears its buffered text. Empty text produces an empty `documents` array. There is no `FLUSH` input in the deployed interface.

`METADATA_IN` must be valid JSON when supplied. It is accumulated into the emitted document metadata; consumers should parse `OUTPUT` rather than assume a fixed document schema from this utility.
