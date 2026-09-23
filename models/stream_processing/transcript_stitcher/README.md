# Transcript Stitcher

Reference Triton Python stage that removes exact normalized word overlap from successive transcript segments. It is used by [GMRS ASR Pipeline](../../ensemble/gmrs_asr_pipeline/README.md).

| Tensor | Direction | Type and shape | Required | Meaning |
| --- | --- | --- | --- | --- |
| `TEXT_IN` | Input | `STRING`, `[1]` | Yes | Current raw transcript segment. |
| `RESET` | Input | `BOOL`, `[1]` | No | Discard the preceding raw transcript before processing. |
| `TEXT_OUT` | Output | `STRING`, `[1]` | Yes | Novel text after overlap removal. |

The model is unbatched and stateful by correlation ID. It retains the previous raw transcript, then compares its final words with the current transcript's initial words using case-folded word tokens. The deployed overlap range is two through 40 words. When it finds no exact overlap, it returns the complete current text.

`RESET=true` clears prior text before the current segment is compared, so the current segment is returned intact and becomes the new reference. This is text-only stitching; it cannot correct transcription differences, punctuation changes that alter tokens, or non-exact overlaps.
