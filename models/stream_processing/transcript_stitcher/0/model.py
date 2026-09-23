# Copyright 2026, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import json
import re

import numpy as np
import triton_python_backend_utils as pb_utils


WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def stitch_transcript(previous, current, min_overlap_words=2, max_overlap_words=40):
    """Return only the novel part of current after an exact normalized overlap."""
    previous = previous.strip()
    current = current.strip()
    if not previous or not current:
        return current

    previous_words = [
        match.group(0).casefold() for match in WORD_RE.finditer(previous)
    ]
    current_matches = list(WORD_RE.finditer(current))
    current_words = [match.group(0).casefold() for match in current_matches]
    maximum = min(len(previous_words), len(current_words), max_overlap_words)

    for count in range(maximum, min_overlap_words - 1, -1):
        if previous_words[-count:] == current_words[:count]:
            novel = current[current_matches[count - 1].end() :]
            return novel.lstrip(" \t\r\n,.;:!?-\u2013\u2014")
    return current


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        self.min_overlap_words = int(
            self._param(params, "min_overlap_words", "2")
        )
        self.max_overlap_words = int(
            self._param(params, "max_overlap_words", "40")
        )
        if self.min_overlap_words < 1:
            raise pb_utils.TritonModelException(
                "min_overlap_words must be at least one"
            )
        if self.max_overlap_words < self.min_overlap_words:
            raise pb_utils.TritonModelException(
                "max_overlap_words must be at least min_overlap_words"
            )
        self.previous = {}

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                stream_id = request.correlation_id()
                if self._reset_requested(request):
                    self.previous.pop(stream_id, None)

                tensor = pb_utils.get_input_tensor_by_name(request, "TEXT_IN")
                if tensor is None:
                    raise ValueError("Missing required input TEXT_IN")
                values = tensor.as_numpy().reshape(-1)
                current = self._to_str(values[0]) if values.size else ""
                output = stitch_transcript(
                    self.previous.get(stream_id, ""),
                    current,
                    self.min_overlap_words,
                    self.max_overlap_words,
                )
                self.previous[stream_id] = current

                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "TEXT_OUT", np.array([output], dtype=object)
                            )
                        ]
                    )
                )
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
                )
        return responses

    @staticmethod
    def _param(params, key, default):
        return params.get(key, {}).get("string_value", default)

    @staticmethod
    def _reset_requested(request):
        tensor = pb_utils.get_input_tensor_by_name(request, "RESET")
        if tensor is None:
            return False
        values = tensor.as_numpy().reshape(-1)
        return values.size > 0 and bool(values[0])

    @staticmethod
    def _to_str(value):
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return str(value)
