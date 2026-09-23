# Copyright 2026, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import json
import uuid
from datetime import datetime, timezone

import numpy as np
import triton_python_backend_utils as pb_utils

EXCLUDED_SIGMF_FIELDS = {"core:extensions"}


class SequenceState:
    def __init__(self):
        self.buffer = []
        self.char_count = 0
        self.doc_index = 0
        self.uuid = str(uuid.uuid4())
        self.metadata = {}

    def add_metadata(self, metadata):
        if metadata:
            self.metadata = strip_sigmf_fields(metadata)

    def add_text(self, text):
        text = text.strip()
        if text:
            self.buffer.append(text)
            self.char_count += len(text)

    def reset(self):
        self.buffer = []
        self.char_count = 0
        self.doc_index += 1

    def build_document(self):
        now = datetime.now(timezone.utc)
        now_str = now.isoformat().replace("+00:00", "Z")
        transcript = " ".join(self.buffer).strip()
        metadata = self._doc_metadata(now_str)

        return {
            "document": transcript,
            "doc_index": self.doc_index,
            "doc_metadata": metadata,
        }

    def _doc_metadata(self, now_str):
        global_meta = self.metadata.get("global", {})
        captures = self.metadata.get("captures", [])
        capture0 = captures[0] if captures else {}

        frequency = capture0.get("core:frequency", 0)
        hostname = global_meta.get("core:author", "")
        stream_id = f"gmrs-{frequency}" if frequency else "gmrs"

        return {
            "streamId": stream_id,
            "doc_id": f"{hostname}-{stream_id}" if hostname else stream_id,
            "timestamp": now_str,
            "radio_metadata": self.metadata,
        }


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})
        self.min_chars = int(self._param(params, "min_chars_per_batch", "0"))
        self.max_chars = int(self._param(params, "max_chars_per_batch", "0"))
        self.states = {}

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                state = self._state_for(request)
                metadata = self._metadata_from_request(request)
                state.add_metadata(metadata)

                text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT_IN")
                if text_tensor is None:
                    raise ValueError("Missing required input TEXT_IN")
                text = self._to_str(text_tensor.as_numpy()[0])
                state.add_text(text)

                documents = []
                if self._should_emit(state):
                    documents.append(state.build_document())
                    state.reset()

                payload = json.dumps({"documents": documents})
                out = pb_utils.Tensor("OUTPUT", np.array([payload], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
                )
        return responses

    def _state_for(self, request):
        corr_id = request.correlation_id()
        if corr_id not in self.states:
            self.states[corr_id] = SequenceState()
        return self.states[corr_id]

    def _should_emit(self, state):
        if state.char_count == 0:
            return False
        if self.max_chars > 0 and state.char_count >= self.max_chars:
            return True
        if self.min_chars <= 0:
            return True
        return state.char_count >= self.min_chars

    def _metadata_from_request(self, request):
        tensor = pb_utils.get_input_tensor_by_name(request, "METADATA_IN")
        if tensor is None:
            return {}
        values = tensor.as_numpy().reshape(-1)
        if values.size == 0:
            return {}
        text = self._to_str(values[0])
        if not text:
            return {}
        return json.loads(text)

    @staticmethod
    def _param(params, key, default):
        return params.get(key, {}).get("string_value", default)

    @staticmethod
    def _to_str(value):
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return str(value)


def strip_sigmf_fields(value):
    if isinstance(value, dict):
        return {
            key: strip_sigmf_fields(item)
            for key, item in value.items()
            if key not in EXCLUDED_SIGMF_FIELDS
        }
    if isinstance(value, list):
        return [strip_sigmf_fields(item) for item in value]
    return value
