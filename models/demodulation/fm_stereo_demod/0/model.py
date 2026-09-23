# Copyright 2026, Deepwave Digital, Inc.
# SPDX-License-Identifier: GPL-3.0

import json
import math
import threading
import time

import numpy as np
import triton_python_backend_utils as pb_utils
from gnuradio import analog, blocks, filter, gr
from gnuradio.filter import firdes, window

def deemph_iir_taps(fs, tau_us=75.0):
    tau = tau_us * 1e-6
    alpha = math.exp(-1.0 / (fs * tau))
    return [1.0 - alpha], [1.0, -alpha]


class NumpySource(gr.basic_block):
    def __init__(self):
        gr.basic_block.__init__(self, name="NumpySource", in_sig=None, out_sig=[np.complex64])
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
            #print(f"eos: {self.eos} pending samples: {self.pending_samples}")
            if self.eos and self.pending_samples == 0:
                self.queue = []
                self.eos = False
                self.batch_done_event.set()
                return 0 
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

    def wait_until_idle(self, quiet_seconds=0.05, timeout_seconds=5.0):
        """Wait for downstream filters to finish after the source is drained."""
        deadline = time.monotonic() + timeout_seconds
        with self.condition:
            observed_len = len(self.buffer)
            quiet_deadline = time.monotonic() + quiet_seconds
            while True:
                now = time.monotonic()
                if now >= quiet_deadline or now >= deadline:
                    return
                self.condition.wait(timeout=min(quiet_deadline, deadline) - now)
                current_len = len(self.buffer)
                if current_len != observed_len:
                    observed_len = current_len
                    quiet_deadline = time.monotonic() + quiet_seconds


class StereoFMDemodTensor(gr.top_block):
    """Stereo FM demodulator that takes a numpy IQ tensor and returns audio as a numpy tensor.

    Can be instantiated once and reused by updating the input IQ vector between runs.
    """

    def __init__(self,
                 samp_rate: float = 2.4e6,
                 audio_rate: float = 16000.0,
                 chan_bw: float = 192e3,
                 transition: float = 24e3,
                 station_offset: float = 0.0,
                 deviation: float = 75e3,
                 deemph_us: float = 75.0,
                 demod_rate: float = 250e3):
        gr.top_block.__init__(self, "Stereo FM Demod Tensor")

        # Store params for potential reuse checks
        self._params = {
            "samp_rate": samp_rate,
            "audio_rate": audio_rate,
            "chan_bw": chan_bw,
            "transition": transition,
            "station_offset": station_offset,
            "deviation": deviation,
            "deemph_us": deemph_us,
            "demod_rate": demod_rate,
        }

        # Choose an integer decimation near the requested FM demodulation rate.
        # At 1.92 MS/s and a 250 kHz target this produces a 240 kHz IF, leaving
        # enough bandwidth for a broadcast-FM channel.
        decim = max(1, int(round(samp_rate / demod_rate)))
        base_rate = samp_rate / decim
        base_rate_i = int(round(base_rate))

        # --- Vector source with complex IQ (can be updated via set_iq) ---
        self.src = NumpySource()
        self.snk = SyncSink()

        # --- Channelize and translate ---
        chan_taps = firdes.low_pass(1.0, samp_rate, chan_bw/2, transition, window.WIN_HAMMING)
        self.tuner = filter.freq_xlating_fir_filter_ccf(decim, chan_taps, station_offset, samp_rate)

        # --- Squelch noise when no transmitter/speaker
        self.squelch = analog.pwr_squelch_cc(
            db=-58.0,   # dBFS-ish; find via trial (e.g., -60 to -40)
            alpha=1e-3,      # smoothing
            ramp=0,
            gate=True        # mute instead of ramping
        )

        # --- Automatic Gain Control BEFORE demod (complex->complex)
        self.agc = analog.agc2_cc(
            attack_rate=2e-3,
            decay_rate=6e-5,
            reference=0.8,
            gain=1.0,
        )

        # --- FM discriminator ---
        k = base_rate / (2.0 * math.pi * deviation)
        self.fm_discriminator = analog.quadrature_demod_cf(k)
        self.dc_block = filter.dc_blocker_ff(128, True)
        self.composite_level = blocks.multiply_const_ff(1.5)

        # L+R path
        self.lp_lpr = filter.fir_filter_fff(1, firdes.low_pass(1.0, base_rate, 15e3, 3e3, window.WIN_HAMMING))

        # Pilot -> PLL -> 38 kHz ref
        self.bp_pilot = filter.fir_filter_fff(1, firdes.band_pass(1.0, base_rate, 18.5e3, 19.5e3, 1.0e3, window.WIN_HAMMING))
        self.float_to_cplx = blocks.float_to_complex(1)
        fmin = 18.9e3 / base_rate * 2*math.pi
        fmax = 19.1e3 / base_rate * 2*math.pi
        self.pll = analog.pll_refout_cc(2*math.pi*50/base_rate, fmin, fmax)
        self.square = blocks.multiply_cc()
        self.ref38 = blocks.complex_to_real(1)

        # L-R DSB-SC path
        self.bp_lmr = filter.fir_filter_fff(1, firdes.band_pass(1.0, base_rate, 23e3, 53e3, 3e3, window.WIN_HAMMING))
        self.mult_lmr = blocks.multiply_ff()
        self.lp_lmr = filter.fir_filter_fff(1, firdes.low_pass(1.0, base_rate, 15e3, 3e3, window.WIN_HAMMING))

        # Stereo matrix
        self.add = blocks.add_ff()
        self.sub = blocks.sub_ff()
        self.scaleL = blocks.multiply_const_ff(0.5)
        self.scaleR = blocks.multiply_const_ff(0.5)

        # De-emphasis runs before the audio resampler, so its coefficients must
        # be designed for the FM intermediate rate rather than the output rate.
        b, a = deemph_iir_taps(base_rate, deemph_us)
        self.deemp_L = filter.iir_filter_ffd(b, a)
        self.deemp_R = filter.iir_filter_ffd(b, a)

        # Audio low-pass
        self.audio_lpf_L = filter.fir_filter_fff(1, firdes.low_pass_2(1.0, base_rate, 8e3, 2e3, 18.0))
        self.audio_lpf_R = filter.fir_filter_fff(1, firdes.low_pass_2(1.0, base_rate, 8e3, 2e3, 18.0))

        # Rational resampler with reduced ratio
        g = math.gcd(int(audio_rate), base_rate_i)
        interp = int(audio_rate) // g
        decim = base_rate_i // g
        self.resampL = filter.rational_resampler_fff(interpolation=interp, decimation=decim)
        self.resampR = filter.rational_resampler_fff(interpolation=interp, decimation=decim)

        # Interleave stereo and capture to a tensor sink
        #self.interleave = blocks.interleave_ff()
        self.interleave = blocks.interleave(gr.sizeof_float*1, 2)

        # --- Wire graph ---
        self.connect(self.src, self.tuner)
        self.connect(self.tuner, self.squelch)
        self.connect(self.squelch, self.agc)
        #self.connect(self.tuner, self.agc)
        self.connect(self.agc, self.fm_discriminator)
        self.connect(self.fm_discriminator, self.dc_block)
        self.connect(self.dc_block, self.composite_level)

        # L+R
        self.connect(self.composite_level, self.lp_lpr)

        # Pilot PLL to 38 kHz
        self.connect(self.composite_level, self.bp_pilot)
        self.connect(self.bp_pilot, self.float_to_cplx)
        self.connect((self.float_to_cplx, 0), (self.pll, 0))
        self.connect(self.pll, (self.square, 0))
        self.connect(self.pll, (self.square, 1))
        self.connect(self.square, self.ref38)

        # L-R recovery
        self.connect(self.composite_level, self.bp_lmr)
        self.connect(self.bp_lmr, (self.mult_lmr, 0))
        self.connect(self.ref38, (self.mult_lmr, 1))
        self.connect(self.mult_lmr, self.lp_lmr)

        # Matrix -> de-emphasis -> resample
        self.connect(self.lp_lpr, (self.add, 0))
        self.connect(self.lp_lmr, (self.add, 1))
        self.connect(self.add, self.scaleL)
        self.connect(self.scaleL, self.deemp_L)
        self.connect(self.deemp_L, self.audio_lpf_L)
        self.connect(self.audio_lpf_L, self.resampL)

        self.connect(self.lp_lpr, (self.sub, 0))
        self.connect(self.lp_lmr, (self.sub, 1))
        self.connect(self.sub, self.scaleR)
        self.connect(self.scaleR, self.deemp_R)
        self.connect(self.deemp_R, self.audio_lpf_R)
        self.connect(self.audio_lpf_R, self.resampR)

        self.connect(self.resampL, (self.interleave, 0))
        self.connect(self.resampR, (self.interleave, 1))
        self.connect(self.interleave, self.snk)

    def process(self, iq_complex):
        self.snk.reset()
        self.src.push(iq_complex)
        self.src.batch_done_event.wait()
        self.snk.wait_until_idle()
        out = self.snk.get_data()
        if out.size % 2 != 0:
            out = out[:-1]
        stereo = out.reshape(-1, 2)
        return stereo[:, 0] + stereo[:,1]


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

        # Default parameter settings
        self.sample_rate = float(self.parameters.get(
            "sample_rate", {}).get("string_value", "1920000"))
        self.audio_rate = float(self.parameters.get(
            "audio_rate", {}).get("string_value", "16000"))
        self.demod_rate = float(self.parameters.get(
            "demod_rate", {}).get("string_value", "250000"))
        if self.sample_rate <= 0 or self.audio_rate <= 0 or self.demod_rate <= 0:
            raise pb_utils.TritonModelException(
                "sample_rate, audio_rate, and demod_rate must be positive"
            )


        self.demod = StereoFMDemodTensor(
            samp_rate=self.sample_rate,
            audio_rate=self.audio_rate,
            demod_rate=self.demod_rate,
        )
        self.demod.start()

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "AUDIO_OUT")
        
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
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "IQ_IN")
            metadata_tensor = pb_utils.get_input_tensor_by_name(request, "META_IN")
            in_metadata = (
                metadata_tensor.as_numpy()
                if metadata_tensor is not None
                else np.array([""], dtype=object)
            )
            in_complex = in_0.as_numpy().view(np.complex64)
            out_0 = self.demod.process(in_complex)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor(
                "AUDIO_OUT", out_0.astype(output0_dtype))
            

            out_metadata = pb_utils.Tensor("META_OUT", in_metadata)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_metadata]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass
