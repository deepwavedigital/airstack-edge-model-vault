# Copyright 2025 Mistral AI
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Deepwave Digital, Inc. in 2026.

import json
import os
import sys
import threading
import time

import soxr

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.parameters = self.model_config.get("parameters", {})

        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = self._param("model_path", model_dir)
        self.device = self._param("device", "cuda")
        self.max_tokens = int(self._param("max_tokens", "256"))
        self.input_sample_rate = int(self._param("input_sample_rate", "16000"))
        self.max_audio_seconds = float(self._param("max_audio_seconds", "30.0"))
        warmup_seconds = float(self._param("warmup_seconds", "1.0"))
        compile_model = self._bool_param("compile", False)
        warmup_enabled = self._bool_param("warmup", True)
        upstream_verbose = self._bool_param("upstream_verbose", False)
        page_cache_policy = self._param("page_cache_policy", "startup").lower()

        if self.max_tokens <= 0:
            raise pb_utils.TritonModelException("max_tokens must be positive")
        if self.input_sample_rate <= 0:
            raise pb_utils.TritonModelException(
                "input_sample_rate must be positive"
            )
        if self.max_audio_seconds <= 0:
            raise pb_utils.TritonModelException(
                "max_audio_seconds must be positive"
            )

        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        from voxtral_runtime import VoxtralRuntime

        load_started = time.perf_counter()
        self.runtime = VoxtralRuntime(
            self.model_path,
            device=self.device,
            compile_model=compile_model,
            page_cache_policy=page_cache_policy,
            verbose=upstream_verbose,
        )
        self.sample_rate = self.runtime.sample_rate
        self._inference_lock = threading.Lock()

        if warmup_enabled:
            self.runtime.warmup(warmup_seconds)

        pb_utils.Logger.log_info(
            "Voxtral loaded and ready in "
            f"{time.perf_counter() - load_started:.2f} seconds"
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_IN")
                if input_tensor is None:
                    raise ValueError("Missing required input AUDIO_IN")
                raw_audio = input_tensor.as_numpy()
                audio = self._prepare_audio(raw_audio)
                if audio.size == 0:
                    responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[
                                pb_utils.Tensor(
                                    "TEXT_OUT", np.array([""], dtype=object)
                                )
                            ]
                        )
                    )
                    continue

                duration_seconds = audio.size / self.sample_rate
                if duration_seconds > self.max_audio_seconds:
                    raise ValueError(
                        f"Audio duration {duration_seconds:.2f}s exceeds configured "
                        f"maximum {self.max_audio_seconds:.2f}s"
                    )

                audio_rms = float(
                    np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
                )
                audio_peak = float(np.max(np.abs(audio)))

                started = time.perf_counter()
                with self._inference_lock:
                    text = self.runtime.transcribe(
                        audio,
                        max_tokens=self.max_tokens,
                    )
                pb_utils.Logger.log_info(
                    f"Transcribed {duration_seconds:.2f}s of audio "
                    f"(rms={audio_rms:.6f}, peak={audio_peak:.6f}) in "
                    f"{time.perf_counter() - started:.3f}s; "
                    f"produced {len((text or '').strip())} characters"
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "TEXT_OUT", np.array([text or ""], dtype=object)
                            )
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

    def _param(self, key: str, default: str) -> str:
        return self.parameters.get(key, {}).get("string_value", default)

    def _bool_param(self, key: str, default: bool) -> bool:
        value = self._param(key, str(default)).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise pb_utils.TritonModelException(
            f"Parameter {key!r} must be true or false, got {value!r}"
        )

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        if audio.size and self.input_sample_rate != self.sample_rate:
            audio = soxr.resample(
                audio, self.input_sample_rate, self.sample_rate, quality="HQ"
            )

        return np.ascontiguousarray(audio, dtype=np.float32)

    def finalize(self):
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            runtime.close()
            self.runtime = None
