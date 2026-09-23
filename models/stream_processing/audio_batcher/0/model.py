# Copyright 2026, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import json

import numpy as np
import triton_python_backend_utils as pb_utils


class AudioState:
    def __init__(self):
        self.buffer = np.zeros(0, dtype=np.float32)


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})
        self.sample_rate = int(self._param(params, "sample_rate", "16000"))
        self.chunk_samples = self._seconds_to_samples(
            self._param(params, "chunk_seconds", "10.0")
        )
        self.overlap_samples = self._seconds_to_samples(
            self._param(params, "overlap_seconds", "2.0")
        )
        self.max_buffer_samples = self._seconds_to_samples(
            self._param(params, "max_buffer_seconds", "60.0")
        )
        self.states = {}

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                state = self._state_for(request)
                tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_IN")
                if tensor is None:
                    raise ValueError("Missing required input AUDIO_IN")

                audio = np.asarray(tensor.as_numpy(), dtype=np.float32).reshape(-1)
                audio = self._sanitize_audio(audio)
                output = self._audio_for_request(state, request, audio)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("AUDIO_OUT", output.astype(np.float32))
                        ]
                    )
                )
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(str(exc)),
                    )
                )
        return responses

    def _audio_for_request(
        self, state: AudioState, request, audio: np.ndarray
    ) -> np.ndarray:
        if self.chunk_samples <= 0:
            return audio

        if audio.size:
            if state.buffer.size:
                state.buffer = np.concatenate([state.buffer, audio])
            else:
                state.buffer = audio.copy()

        if (
            self.max_buffer_samples > 0
            and state.buffer.size > self.max_buffer_samples
        ):
            state.buffer = state.buffer[-self.max_buffer_samples :]

        flush = self._flush_from_request(request)
        if state.buffer.size < self.chunk_samples and not flush:
            return np.zeros(0, dtype=np.float32)

        if flush:
            output = state.buffer.copy()
            state.buffer = np.zeros(0, dtype=np.float32)
            return output

        output = state.buffer[: self.chunk_samples].copy()
        advance = self.chunk_samples - self.overlap_samples
        if advance <= 0:
            advance = self.chunk_samples
        state.buffer = state.buffer[advance:].copy()
        return output

    def _state_for(self, request) -> AudioState:
        corr_id = request.correlation_id()
        if corr_id not in self.states:
            self.states[corr_id] = AudioState()
        return self.states[corr_id]

    def _flush_from_request(self, request) -> bool:
        tensor = pb_utils.get_input_tensor_by_name(request, "FLUSH")
        if tensor is None:
            return False
        values = tensor.as_numpy().reshape(-1)
        return values.size > 0 and bool(values[0])

    def _seconds_to_samples(self, value: str) -> int:
        return max(0, int(round(float(value) * self.sample_rate)))

    @staticmethod
    def _param(params, key: str, default: str) -> str:
        return params.get(key, {}).get("string_value", default)

    @staticmethod
    def _sanitize_audio(audio: np.ndarray) -> np.ndarray:
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(audio, dtype=np.float32)
