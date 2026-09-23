# Copyright 2025 Mistral AI
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Deepwave Digital, Inc. in 2026.

"""Serving-only adapter for the upstream Jetson Voxtral implementation."""

import contextlib
import gc
import logging
import os
import sys
from typing import Optional

import numpy as np


LOGGER = logging.getLogger("voxtral-mini-stt.runtime")


class VoxtralRuntime:
    """Own one resident Voxtral model and hide upstream server concerns.

    The downloaded ``jetson_serve_sdpa.py`` also contains a CLI and WebSocket
    server. Importing its model class does not start either server, which lets
    Triton remain the only process-level serving interface.
    """

    VALID_PAGE_CACHE_POLICIES = {"always", "startup", "never"}

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        compile_model: bool = False,
        page_cache_policy: str = "startup",
        verbose: bool = False,
    ):
        if page_cache_policy not in self.VALID_PAGE_CACHE_POLICIES:
            choices = ", ".join(sorted(self.VALID_PAGE_CACHE_POLICIES))
            raise ValueError(
                f"Invalid page_cache_policy {page_cache_policy!r}; use {choices}"
            )

        self.model_path = model_path
        self.device = device
        self.page_cache_policy = page_cache_policy
        self.verbose = verbose
        self._model = None
        self._torch = None
        self._devnull = None
        self._page_cache_warning_logged = False
        self.sample_rate: Optional[int] = None

        self._validate_artifacts()
        self._weights_path = os.path.join(
            self.model_path, "consolidated.safetensors"
        )
        scripts_dir = os.path.join(model_path, "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from jetson_serve_sdpa import SAMPLE_RATE, VoxtralModel

        import torch

        self._torch = torch
        self.sample_rate = int(SAMPLE_RATE)
        # Upstream reclaims page cache by allocating and zeroing up to 4 GiB.
        # The kernel may OOM-kill Triton before that raises MemoryError. Model
        # construction invokes the method, so override it before loading starts.
        reclaim_page_cache = self._reclaim_weight_page_cache

        class ServingVoxtralModel(VoxtralModel):
            @staticmethod
            def _evict_cache():
                reclaim_page_cache()

        self._model = ServingVoxtralModel(
            model_path,
            device=device,
            compile=compile_model,
        )
        self._model._load_tokenizer()
        self._configure_page_cache_policy()

        if not self.verbose:
            self._devnull = open(os.devnull, "w", encoding="utf-8")

    def _reclaim_weight_page_cache(self) -> None:
        """Discard cached weight pages without allocating memory to pressure RAM."""
        if not hasattr(os, "posix_fadvise") or not hasattr(
            os, "POSIX_FADV_DONTNEED"
        ):
            if not self._page_cache_warning_logged:
                LOGGER.warning(
                    "posix_fadvise is unavailable; skipping weight page-cache reclaim"
                )
                self._page_cache_warning_logged = True
            return

        try:
            fd = os.open(self._weights_path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError as exc:
            if not self._page_cache_warning_logged:
                LOGGER.warning("Could not reclaim weight page cache: %s", exc)
                self._page_cache_warning_logged = True

    def _validate_artifacts(self) -> None:
        required = (
            "consolidated.safetensors",
            "params.json",
            "tekken.json",
            os.path.join("scripts", "jetson_serve_sdpa.py"),
        )
        missing = [
            os.path.join(self.model_path, item)
            for item in required
            if not os.path.exists(os.path.join(self.model_path, item))
        ]
        if missing:
            formatted = "\n  - ".join(missing)
            raise RuntimeError(f"Missing Voxtral artifacts:\n  - {formatted}")

    def _configure_page_cache_policy(self) -> None:
        if self.page_cache_policy == "always":
            LOGGER.info("Page-cache reclaim enabled before every inference")
            return

        if not callable(getattr(self._model, "_evict_cache", None)):
            raise RuntimeError(
                "Upstream Voxtral model does not expose the expected "
                "_evict_cache method"
            )

        if self.page_cache_policy == "startup":
            LOGGER.info("Reclaiming page cache once after model loading")
            self._model._evict_cache()

        # The upstream transcribe() calls this method on every request. Replace
        # that call after startup while retaining the original load-time logic.
        self._model._evict_cache = lambda: None
        LOGGER.info("Per-request page-cache reclaim disabled")

    def warmup(self, audio_seconds: float = 1.0) -> None:
        """Finish lazy CUDA setup before Triton marks the model ready."""
        if audio_seconds <= 0:
            return

        samples = max(1, int(round(self.sample_rate * audio_seconds)))
        silence = np.zeros(samples, dtype=np.float32)
        LOGGER.info("Warming Voxtral with %.2f seconds of audio", audio_seconds)
        try:
            with self._output_context():
                self._model.transcribe(silence, max_tokens=8)
            if self._torch.cuda.is_available():
                self._torch.cuda.synchronize()
        finally:
            self._release_transient_memory()
        LOGGER.info("Voxtral warm-up complete")

    def transcribe(self, audio: np.ndarray, *, max_tokens: int) -> str:
        if self._model is None:
            raise RuntimeError("Voxtral runtime is closed")
        try:
            with self._output_context():
                return self._model.transcribe(audio, max_tokens=max_tokens) or ""
        finally:
            self._release_transient_memory()

    def _release_transient_memory(self) -> None:
        # Jetson CUDA and CPU share physical memory. PyTorch's caching allocator
        # otherwise retains transient encoder, KV-cache, and logits allocations,
        # leaving too little headroom for Triton's own pinned and CUDA pools.
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _output_context(self):
        if self.verbose:
            return contextlib.nullcontext()
        return contextlib.redirect_stdout(self._devnull)

    def close(self) -> None:
        if self._model is None:
            return
        self._model = None
        if self._devnull is not None:
            self._devnull.close()
            self._devnull = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
