# Copyright 2026, Deepwave Digital, Inc.
# SPDX-License-Identifier: GPL-3.0

import json
import numpy as np
import triton_python_backend_utils as pb_utils
from gnuradio import gr, filter, analog
from gnuradio.filter import firdes, window
import threading
import sys
import logging
import time

logging.root.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class NumpySource(gr.basic_block):
    def __init__(self):
        gr.basic_block.__init__(
            self, name="NumpySource", in_sig=None, out_sig=[np.complex64]
        )
        self.queue = []
        self.lock = threading.Lock()
        self.pending_samples = 0
        self.batch_done_event = threading.Event()
        self.eos = False

    def push(self, arr: np.ndarray):
        with self.lock:
            self.queue.append(arr)
            self.pending_samples = len(arr)
            self.eos = True
            self.batch_done_event.clear()

    def general_work(self, input_items, output_items):
        out = output_items[0]
        produced = 0
        with self.lock:
            if self.queue:
                data = self.queue.pop(0)
                n = min(len(data), len(out))
                out[:n] = data[:n]
                produced = n
                self.pending_samples -= n
                if n < len(data):
                    self.queue.insert(0, data[n:])
            if self.eos and self.pending_samples == 0:
                self.queue = []
                self.eos = False
                self.batch_done_event.set()
                # The return value tells GNU Radio how many samples from this
                # invocation are valid. Returning zero here drops the final
                # chunk (and drops the entire request when it fits in one
                # scheduler buffer).
        return produced


class SyncSink(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(self, name="SyncSink", in_sig=[np.float32], out_sig=None)
        self.buffer = []
        self.condition = threading.Condition()

    def reset(self):
        with self.condition:
            self.buffer = []

    def work(self, input_items, output_items):
        data = input_items[0]
        if len(data) > 0:
            with self.condition:
                self.buffer.extend(data)
                self.condition.notify_all()
        return len(data)

    def get_data(self):
        with self.condition:
            return np.array(self.buffer, dtype=np.float32)

    def wait_until_idle(self, quiet_seconds=0.05, timeout_seconds=30.0):
        """Wait until downstream blocks stop delivering samples."""
        deadline = time.monotonic() + timeout_seconds
        with self.condition:
            observed_len = len(self.buffer)
            quiet_deadline = time.monotonic() + quiet_seconds
            while True:
                now = time.monotonic()
                if now >= quiet_deadline:
                    return
                if now >= deadline:
                    raise TimeoutError("Timed out waiting for demodulated audio")
                self.condition.wait(timeout=min(quiet_deadline, deadline) - now)
                current_len = len(self.buffer)
                if current_len != observed_len:
                    observed_len = current_len
                    quiet_deadline = time.monotonic() + quiet_seconds


class GMRSFMDemodTensor(gr.top_block):
    """GMRS/NBFM demodulator that takes a numpy IQ tensor and returns audio as a numpy tensor."""

    def __init__(
        self,
        samp_rate: float = 1.92e6,
        audio_rate: int = 48000,
        dec1: int = 20,
        dec2: int = 5,
        lpf_cutoff: float = 6e3,
        lpf_transition: float = 2e3,
        squelch_db: float = -80.0,
        squelch_alpha: float = 1e-2,
        max_dev: float = 5e3,
        deemph_us: float = 750.0,
    ):
        gr.top_block.__init__(self, "GMRS NBFM Demod Tensor")

        if samp_rate <= 0 or dec1 <= 0 or dec2 <= 0:
            raise ValueError("samp_rate, dec1, and dec2 must be positive")
        if audio_rate is None or audio_rate <= 0:
            audio_rate = int(samp_rate / dec1 / dec2)
        audio_rate = int(audio_rate)
        if audio_rate <= 0:
            raise ValueError("audio_rate must be positive")
        quad_rate_float = samp_rate / dec1
        quad_rate = int(round(quad_rate_float))
        if not np.isclose(quad_rate_float, quad_rate):
            raise ValueError("samp_rate must be evenly divisible by dec1")
        if quad_rate % audio_rate:
            raise ValueError(
                "The post-channelization rate must be an integer multiple "
                "of audio_rate"
            )
        if quad_rate // audio_rate != dec2:
            raise ValueError("dec2 must equal (samp_rate / dec1) / audio_rate")

        self._params = {
            "samp_rate": samp_rate,
            "audio_rate": audio_rate,
            "dec1": dec1,
            "dec2": dec2,
            "lpf_cutoff": lpf_cutoff,
            "lpf_transition": lpf_transition,
            "squelch_db": squelch_db,
            "squelch_alpha": squelch_alpha,
            "max_dev": max_dev,
            "deemph_us": deemph_us,
        }

        self.src = NumpySource()
        self.snk = SyncSink()

        self.lpf = filter.fir_filter_ccf(
            dec1,
            firdes.low_pass(
                1.0,
                samp_rate,
                lpf_cutoff,
                lpf_transition,
                window.WIN_HAMMING,
                6.76,
            ),
        )
        self.squelch = analog.pwr_squelch_cc(squelch_db, squelch_alpha, 0, True)
        self.nbfm = analog.nbfm_rx(
            audio_rate=audio_rate,
            quad_rate=quad_rate,
            tau=deemph_us * 1e-6,
            max_dev=max_dev,
        )

        self.connect(self.src, self.lpf)
        self.connect(self.lpf, self.squelch)
        self.connect(self.squelch, self.nbfm)
        self.connect(self.nbfm, self.snk)

    def process(self, iq_complex):
        if iq_complex.size == 0:
            return np.zeros(0, dtype=np.float32)
        self.snk.reset()
        self.src.push(iq_complex)
        if not self.src.batch_done_event.wait(timeout=30.0):
            raise TimeoutError("Timed out waiting for the GMRS IQ source to drain")
        self.snk.wait_until_idle()
        return self.snk.get_data()


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])
        self.parameters = self.model_config.get("parameters", {})

        # Default parameter settings (GMRS/NBFM)
        self.sample_rate = float(
            self.parameters.get("sample_rate", {}).get("string_value", "1920000")
        )
        self.audio_rate = int(
            float(self.parameters.get("audio_rate", {}).get("string_value", "48000"))
        )
        self.dec1 = int(
            float(self.parameters.get("dec1", {}).get("string_value", "20"))
        )
        self.dec2 = int(float(self.parameters.get("dec2", {}).get("string_value", "5")))
        self.lpf_cutoff = float(
            self.parameters.get("lpf_cutoff", {}).get("string_value", "6000")
        )
        self.lpf_transition = float(
            self.parameters.get("lpf_transition", {}).get("string_value", "2000")
        )
        self.squelch_db = float(
            self.parameters.get("squelch_db", {}).get("string_value", "-80")
        )
        self.squelch_alpha = float(
            self.parameters.get("squelch_alpha", {}).get("string_value", ".01")
        )
        self.max_dev = float(
            self.parameters.get("max_dev", {}).get("string_value", "5000")
        )
        self.deemph_us = float(
            self.parameters.get("deemph_us", {}).get("string_value", "750")
        )

        if self.audio_rate <= 0:
            self.audio_rate = int(self.sample_rate / self.dec1 / self.dec2)

        self.demod = GMRSFMDemodTensor(
            samp_rate=self.sample_rate,
            audio_rate=self.audio_rate,
            dec1=self.dec1,
            dec2=self.dec2,
            lpf_cutoff=self.lpf_cutoff,
            lpf_transition=self.lpf_transition,
            squelch_db=self.squelch_db,
            squelch_alpha=self.squelch_alpha,
            max_dev=self.max_dev,
            deemph_us=self.deemph_us,
        )
        self.demod.start()

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "AUDIO_OUT")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype
        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            try:
                in_0 = pb_utils.get_input_tensor_by_name(request, "IQ_IN")
                if in_0 is None:
                    raise ValueError("Missing required input IQ_IN")
                in_metadata_tensor = pb_utils.get_input_tensor_by_name(
                    request, "META_IN"
                )
                if in_metadata_tensor is None:
                    in_metadata = np.array(["{}"], dtype=object)
                else:
                    in_metadata = in_metadata_tensor.as_numpy()

                interleaved_iq = np.asarray(in_0.as_numpy(), dtype=np.float32).reshape(-1)
                if interleaved_iq.size % 2:
                    raise ValueError(
                        "IQ_IN must contain an even number of interleaved I/Q samples"
                    )
                in_complex = np.ascontiguousarray(interleaved_iq).view(np.complex64)
                out_0 = self.demod.process(in_complex)

                out_tensor_0 = pb_utils.Tensor(
                    "AUDIO_OUT", out_0.astype(output0_dtype)
                )
                out_metadata = pb_utils.Tensor("META_OUT", in_metadata)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[out_tensor_0, out_metadata]
                    )
                )
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(exc))
                    )
                )

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        demod = getattr(self, "demod", None)
        if demod is not None:
            demod.stop()
            demod.wait()
            self.demod = None
