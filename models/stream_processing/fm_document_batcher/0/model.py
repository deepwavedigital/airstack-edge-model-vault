# Copyright 2026, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import json
from datetime import datetime, timezone

import numpy as np
import triton_python_backend_utils as pb_utils


SIGMF_FORMATTING_FIELDS = {"core:extensions", "core:version"}


class BatchState:
    def __init__(self):
        self.parts = []
        self.char_count = 0
        self.doc_index = 0
        self.metadata = {}
        self.metadata_fields = {}
        self.start_time = None

    def add_text(self, text, now):
        text = text.strip()
        if text:
            if not self.parts:
                self.start_time = now
            self.parts.append(text)
            self.char_count += len(text)

    def clear_document(self):
        self.parts = []
        self.char_count = 0
        self.doc_index += 1
        self.start_time = None


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        self.min_chars = int(self._param(params, "min_chars_per_batch", "400"))
        self.max_chars = int(self._param(params, "max_chars_per_batch", "2000"))
        self.stream_prefix = self._param(params, "stream_prefix", "fm")
        self.stream_id = self._param(params, "stream_id", "")
        if self.min_chars < 0:
            raise pb_utils.TritonModelException(
                "min_chars_per_batch must be non-negative"
            )
        if self.max_chars < 0:
            raise pb_utils.TritonModelException(
                "max_chars_per_batch must be non-negative"
            )
        if self.max_chars and self.max_chars < self.min_chars:
            raise pb_utils.TritonModelException(
                "max_chars_per_batch must be zero or at least min_chars_per_batch"
            )
        self.states = {}

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            try:
                stream_key = request.correlation_id()
                if self._bool_input(request, "RESET"):
                    self.states.pop(stream_key, None)
                state = self.states.setdefault(stream_key, BatchState())

                metadata = self._metadata_input(request)
                if metadata:
                    state.metadata, state.metadata_fields = self._split_metadata(
                        metadata
                    )

                text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT_IN")
                if text_tensor is None:
                    raise ValueError("Missing required input TEXT_IN")
                text_values = text_tensor.as_numpy().reshape(-1)
                text = self._to_str(text_values[0]) if text_values.size else ""
                state.add_text(text, datetime.now(timezone.utc))

                flush = self._bool_input(request, "FLUSH")
                should_emit = self._should_emit(state, flush)
                pb_utils.Logger.log_info(
                    f"FM document stream {stream_key}: received "
                    f"{len(text.strip())} characters, buffered "
                    f"{state.char_count}/{self.min_chars}, flush={flush}, "
                    f"emit={should_emit}"
                )
                if should_emit:
                    payload = json.dumps(
                        self._build_document(state),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    output = np.array([payload], dtype=object)
                    state.clear_document()
                    response = pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("OUTPUT", output)]
                    )
                    response_sender.send(
                        response,
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                    )
                else:
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
            except Exception as exc:
                response_sender.send(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(exc))
                    ),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
        return None

    def _should_emit(self, state, flush):
        if state.char_count == 0:
            return False
        if flush:
            return True
        if self.max_chars and state.char_count >= self.max_chars:
            return True
        return self.min_chars == 0 or state.char_count >= self.min_chars

    def _build_document(self, state):
        now = datetime.now(timezone.utc)
        start_time = state.start_time or now
        fields = state.metadata_fields
        captures = state.metadata.get("captures", [])
        capture = captures[0] if captures else {}
        frequency = capture.get("core:frequency", 0)
        frequency_text = self._frequency_text(frequency)
        generated_stream_id = (
            f"{self.stream_prefix}-{frequency_text}"
            if frequency_text
            else self.stream_prefix
        )
        stream_id = fields.get("streamId") or self.stream_id or generated_stream_id
        start_ntp_float = self._timestamp_float(
            fields.get("start_ntp_float"), fields.get("start_ntp"), start_time
        )
        end_ntp_float = self._timestamp_float(
            fields.get("end_ntp_float"), fields.get("end_ntp"), now
        )

        return {
            "document": " ".join(state.parts).strip(),
            "doc_index": state.doc_index,
            "doc_metadata": {
                "streamId": stream_id,
                "start_ntp_float": start_ntp_float,
                "end_ntp_float": end_ntp_float,
                "radio_metadata": state.metadata,
            },
        }

    @staticmethod
    def _split_metadata(metadata):
        """Accept raw SigMF metadata or an enriched document metadata object."""
        fields = metadata.get("doc_metadata", metadata)
        radio_metadata = fields.get("radio_metadata")
        if isinstance(radio_metadata, dict):
            return sanitize_radio_metadata(radio_metadata), fields
        return sanitize_radio_metadata(metadata), fields

    def _metadata_input(self, request):
        tensor = pb_utils.get_input_tensor_by_name(request, "METADATA_IN")
        if tensor is None:
            return {}
        values = tensor.as_numpy().reshape(-1)
        if not values.size:
            return {}
        value = self._to_str(values[0]).strip()
        return json.loads(value) if value else {}

    @staticmethod
    def _bool_input(request, name):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return False
        values = tensor.as_numpy().reshape(-1)
        return values.size > 0 and bool(values[0])

    @staticmethod
    def _frequency_text(value):
        if value in (None, "", 0):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    @staticmethod
    def _timestamp_float(value, timestamp, fallback):
        if value not in (None, ""):
            return float(value)
        if timestamp not in (None, ""):
            try:
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return parsed.timestamp()
            except (TypeError, ValueError):
                pass
        return fallback.timestamp()

    @staticmethod
    def _param(params, key, default):
        return params.get(key, {}).get("string_value", default)

    @staticmethod
    def _to_str(value):
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return str(value)


def sanitize_radio_metadata(value):
    """Remove SigMF declarations and empty containers, retaining radio state."""
    if isinstance(value, dict):
        cleaned = {
            key: sanitize_radio_metadata(item)
            for key, item in value.items()
            if key not in SIGMF_FORMATTING_FIELDS
        }
        return {
            key: item
            for key, item in cleaned.items()
            if item not in (None, "", [], {})
        }
    if isinstance(value, list):
        cleaned = [sanitize_radio_metadata(item) for item in value]
        return [item for item in cleaned if item not in (None, "", [], {})]
    return value
