# Copyright 2026, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        sample_rate = int(self._param(params, "sample_rate", "16000"))
        overlap_seconds = float(self._param(params, "overlap_seconds", "2.0"))
        if sample_rate <= 0:
            raise pb_utils.TritonModelException("sample_rate must be positive")
        if overlap_seconds < 0:
            raise pb_utils.TritonModelException(
                "overlap_seconds must be non-negative"
            )
        self.overlap_samples = int(round(sample_rate * overlap_seconds))
        self.tails = {}

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_IN")
                if tensor is None:
                    raise ValueError("Missing required input AUDIO_IN")

                stream_id = request.correlation_id()
                if self._reset_requested(request):
                    self.tails.pop(stream_id, None)

                audio = np.asarray(tensor.as_numpy(), dtype=np.float32).reshape(-1)
                audio = self._sanitize_audio(audio)
                tail = self.tails.get(stream_id)
                if tail is not None and tail.size:
                    output = np.concatenate((tail, audio))
                else:
                    output = audio.copy()

                if self.overlap_samples > 0:
                    self.tails[stream_id] = audio[-self.overlap_samples :].copy()
                else:
                    self.tails.pop(stream_id, None)

                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("AUDIO_OUT", output)]
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
    def _sanitize_audio(audio):
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(audio, dtype=np.float32)
