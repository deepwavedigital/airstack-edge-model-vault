#!/usr/bin/env python3
# Copyright 2025 Mistral AI
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Deepwave Digital, Inc. in 2026.

"""Voxtral Mini 4B Realtime — Jetson Orin Nano inference server.

Loads Marlin-packed INT4 weights from consolidated.safetensors and serves
transcription via WebSocket at ws://localhost:8000/v1/realtime.

Key architecture detail: at each decoder position, the input embedding is
  adapter_out[pos] + tok_embed(token_id)
where token_id is BOS/STREAMING_PAD during the prompt, then previously
generated text tokens during generation. Generation runs for exactly
T_adapter total positions.

Memory budget: ~6 GB total (model 4.08 GB + runtime 1.5 GB + KV cache 0.2 GB).

Optimizations over baseline:
  - Marlin fused INT4 dequant+matmul (24x faster generation)
  - F.scaled_dot_product_attention (fused attention kernel)
  - Pre-allocated KV cache (eliminates torch.cat per token per layer)
  - Optional torch.compile (fuses pointwise ops between matmuls)

Usage (inside PyTorch Jetson container):
  pip install safetensors websockets soundfile numpy librosa
  python3 jetson_serve.py --test /workspace/voxtral-quant/test_audio_0.wav
  python3 jetson_serve.py  # starts WebSocket server on port 8000
"""

import argparse
import asyncio
import base64
import json
import math
import os
import sys
import time
import uuid
import warnings
from typing import List, Optional, Tuple

# Suppress harmless warnings
warnings.filterwarnings('ignore', message='.*divide by zero.*')
warnings.filterwarnings('ignore', message='.*FutureWarning.*tokenizer.*')

# Set CUDA allocator config before importing torch (critical for Jetson)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# Try to import Marlin fused INT4 kernel (50x faster than on-the-fly dequantization)
try:
    import marlin as _marlin
    HAS_MARLIN = True
except ImportError:
    HAS_MARLIN = False
    print("WARNING: Marlin not installed. Install with: pip install marlin")

# Try to JIT-compile fused CUDA kernels (collapses ~500 kernel launches/token to ~80)
HAS_FUSED = False
try:
    from torch.utils.cpp_extension import load as _load_ext
    _cu_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'kernels', 'fused_ops.cu')
    if not os.path.exists(_cu_path):
        # Fallback for container mount path
        _cu_path = '/workspace/voxtral-quant/kernels/fused_ops.cu'
    if os.path.exists(_cu_path):
        voxtral_kernels = _load_ext(
            name='voxtral_kernels',
            sources=[_cu_path],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_87'],
            verbose=True
        )
        HAS_FUSED = True
        print(f"Fused CUDA kernels loaded from {_cu_path}")
    else:
        print(f"Fused kernels .cu not found, using PyTorch fallback")
except Exception as e:
    print(f"Fused kernel JIT failed ({e}), using PyTorch fallback")

# ─── Constants ───────────────────────────────────────────────────────────────

BITS = 4
GROUP_SIZE = 128
PACK_FACTOR = 32 // BITS  # 8 int4 values per int32
BIAS = 1 << (BITS - 1)   # 8 (uint4b8 encoding)

TOKEN_BOS = 1
TOKEN_EOS = 2
TOKEN_STREAMING_PAD = 32

SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_MELS = 128
WINDOW_SIZE = 400
GLOBAL_LOG_MEL_MAX = 1.5       # Fixed constant from params.json (NOT per-sample)
RAW_AUDIO_PER_TOK = 1280       # Raw audio samples per adapter token
N_LEFT_PAD_TOKENS = 32         # Left silence padding (aligns with SPAD prompt)
N_RIGHT_PAD_BASE = 17          # Right silence padding (delay+1 + OFFLINE_BUFFER)
DOWNSAMPLE_FACTOR = 4


# ─── Marlin Fused INT4 Linear ────────────────────────────────────────────────

class PrepackedMarlinLinear(nn.Module):
    """Linear layer using pre-packed Marlin INT4 weights from safetensors.

    Loads .B and .s tensors directly — no GPTQ→Marlin conversion needed.
    Used with single-file consolidated.safetensors that already contains
    Marlin-format weights.
    """

    def __init__(self, B, s):
        super().__init__()
        # B: [K//16, 2*N] int32, s: [K//groupsize, N] fp16
        self.in_features = B.shape[0] * 16
        self.out_features = B.shape[1] // 2
        self.register_buffer('B', B)
        self.register_buffer('s', s)
        self.register_buffer('workspace',
                             torch.zeros(self.out_features // 128 * 16,
                                         dtype=torch.int, device=B.device),
                             persistent=False)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        C = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        _marlin.mul(x.view(-1, x.shape[-1]), self.B, C.view(-1, self.out_features),
                    self.s, self.workspace)
        return C




# ─── Building Blocks ─────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x.float() * (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()).type_as(x) * self.weight


class RMSNormFP16(nn.Module):
    """RMSNorm that stays in fp16 — avoids 2x float32 conversion overhead.
    Safe for decoder hidden dim=3072 where values stay in fp16 range.
    Uses fused CUDA kernel when available (single kernel vs 5-6 PyTorch kernels)."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if HAS_FUSED:
            return voxtral_kernels.fused_rmsnorm(x.contiguous(), self.weight, self.eps)
        rms = x.pow(2).mean(-1, keepdim=True).add_(self.eps).rsqrt_()
        return x * rms * self.weight


def apply_rotary_emb(q, k, cos, sin):
    """Interleaved RoPE for Mistral-format Q/K: q,k [B, heads, T, dim], cos,sin [T, dim/2].
    Pairs: (d0,d1), (d2,d3), etc."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_r, q_i = q.float().reshape(*q.shape[:-1], -1, 2).unbind(-1)
    k_r, k_i = k.float().reshape(*k.shape[:-1], -1, 2).unbind(-1)
    return (
        torch.stack([q_r*cos - q_i*sin, q_r*sin + q_i*cos], -1).flatten(-2).type_as(q),
        torch.stack([k_r*cos - k_i*sin, k_r*sin + k_i*cos], -1).flatten(-2).type_as(k),
    )


def apply_rotary_emb_hf(q, k, cos, sin):
    """Half-rotation RoPE for HF-format Q/K in fp16 (avoids float32 conversion).
    q,k: [B, heads, T, dim]. cos,sin: [T, dim/2].
    Uses fused CUDA kernel when available (~14 PyTorch kernels → 2)."""
    if HAS_FUSED:
        return voxtral_kernels.fused_rope_hf(q, k, cos, sin)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    return (
        torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], -1),
        torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], -1),
    )


def make_freqs(dim, maxlen, theta=1e6, device='cpu', dtype=torch.float16):
    f = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(maxlen, device=device)
    a = torch.outer(t, f)
    return a.cos().to(dtype), a.sin().to(dtype)


# ─── KV Cache ────────────────────────────────────────────────────────────────

class KVCache:
    """Pre-allocated KV cache that eliminates per-token torch.cat allocations.

    Instead of creating new tensors and concatenating on every token (3900+
    allocations for a typical transcription), we pre-allocate fixed buffers
    and write into them with an advancing position index.

    Memory cost: 2 * n_layers * n_kv_heads * max_seq * head_dim * 2 bytes
    For 26 layers, 8 heads, 200 seq, 128 dim = ~21 MB (negligible).
    """

    def __init__(self, n_layers, max_seq_len, n_kv_heads, head_dim, device, dtype):
        self.max_seq_len = max_seq_len
        self.pos = 0
        # [n_layers, 1, n_kv_heads, max_seq_len, head_dim]
        self.k = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim,
                             device=device, dtype=dtype)
        self.v = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim,
                             device=device, dtype=dtype)

    def update(self, layer_idx, k_new, v_new):
        """Write new K/V into cache and return full valid slice.

        Args:
            layer_idx: which decoder layer
            k_new, v_new: [1, n_kv_heads, T_new, head_dim] (T_new=prompt_len or 1)

        Returns:
            k_full, v_full: [1, n_kv_heads, pos+T_new, head_dim]
        """
        t_new = k_new.shape[2]
        end = self.pos + t_new
        self.k[layer_idx, :, :, self.pos:end, :] = k_new
        self.v[layer_idx, :, :, self.pos:end, :] = v_new
        return self.k[layer_idx, :, :, :end, :], self.v[layer_idx, :, :, :end, :]

    def advance(self, n):
        """Advance position after all layers have processed n new tokens."""
        self.pos += n

    def reset(self):
        self.pos = 0


# ─── Audio Encoder ───────────────────────────────────────────────────────────

class EncAttn(nn.Module):
    def __init__(self, h=1280, nh=32, hd=64):
        super().__init__()
        self.nh, self.hd = nh, hd
        self.ad = nh * hd
        self.scale = hd ** -0.5
        self.wq = nn.Linear(h, self.ad, bias=True)
        self.wk = nn.Linear(h, self.ad, bias=False)
        self.wv = nn.Linear(h, self.ad, bias=True)
        self.wo = nn.Linear(self.ad, h, bias=True)

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        q, k = apply_rotary_emb(q, k, cos, sin)
        # Encoder uses causal attention with explicit mask
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        return self.wo(a.transpose(1, 2).reshape(B, T, self.ad))


class EncLayer(nn.Module):
    def __init__(self, h=1280, ff=5120):
        super().__init__()
        self.attn = EncAttn(h)
        self.an = RMSNorm(h)
        self.fn = RMSNorm(h)
        self.w1 = nn.Linear(h, ff, bias=False)
        self.w2 = nn.Linear(ff, h, bias=True)
        self.w3 = nn.Linear(h, ff, bias=False)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.an(x), cos, sin, mask)
        h = self.fn(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Encoder(nn.Module):
    def __init__(self, nl=32, h=1280, ff=5120, hd=64):
        super().__init__()
        self.conv1 = nn.Conv1d(N_MELS, h, 3, padding=1)
        self.conv2 = nn.Conv1d(h, h, 3, stride=2, padding=1)
        self.layers = nn.ModuleList([EncLayer(h, ff) for _ in range(nl)])
        self.norm = RMSNorm(h)
        self.hd = hd

    def forward(self, mel):
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        T = x.shape[1]
        cos, sin = make_freqs(self.hd, T, device=x.device, dtype=x.dtype)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device, dtype=x.dtype), 1).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)


# ─── Projector ───────────────────────────────────────────────────────────────

class Projector(nn.Module):
    def __init__(self, inp=5120, out=3072):
        super().__init__()
        self.l1 = nn.Linear(inp, out, bias=False)
        self.l2 = nn.Linear(out, out, bias=False)

    def forward(self, x):
        return self.l2(F.gelu(self.l1(x)))


# ─── LM Decoder Layer ───────────────────────────────────────────────────────

class DecAttn(nn.Module):
    def __init__(self, h=3072, nh=32, nkv=8, hd=128):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.qd, self.kvd = nh*hd, nkv*hd
        self.scale = hd ** -0.5
        self.g = nh // nkv
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None  # set by loader

    def forward(self, x, cos, sin, cache=None, layer_idx=None, is_causal=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rotary_emb_hf(q, k, cos, sin)

        # Update KV cache if available
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        # GQA: expand KV heads to match query heads
        if self.g > 1:
            k = k.repeat_interleave(self.g, 1)
            v = v.repeat_interleave(self.g, 1)

        # Fused attention via SDPA
        # - Prefill (T > 1): use is_causal=True for triangular mask
        # - Decode (T == 1): no mask needed, single query attends to all past
        out = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, is_causal=is_causal)

        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.qd))


class DecLayer(nn.Module):
    def __init__(self, h=3072, ad=32):
        super().__init__()
        self.an = RMSNormFP16(h)
        self.attn = DecAttn()
        self.fn = RMSNormFP16(h)
        self.gate_proj = self.up_proj = self.down_proj = None  # set by loader
        self.ada0 = nn.Linear(h, ad, bias=False)
        self.ada2 = nn.Linear(ad, h, bias=False)
        self._ada_scale = None  # pre-computed: 1 + ada2(gelu(ada0(t_cond)))

    def precompute_ada(self, t_cond):
        """Pre-compute the ada_rms_norm modulation scale from t_cond.
        Since t_cond is constant (delay-based), this eliminates 2 matmuls +
        gelu + add per layer per token (~0.45ms each, 11.7ms total for 26 layers)."""
        with torch.no_grad():
            self._ada_scale = (1.0 + self.ada2(F.gelu(self.ada0(t_cond)))).unsqueeze(0)

    def forward(self, x, cos, sin, cache=None, layer_idx=None,
                is_causal=False, t_cond=None):
        h = self.attn(self.an(x), cos, sin, cache, layer_idx, is_causal)
        x = x + h
        h = self.fn(x)
        if self._ada_scale is not None:
            h = h * self._ada_scale
        elif t_cond is not None:
            h = h * (1.0 + self.ada2(F.gelu(self.ada0(t_cond))).unsqueeze(0))
        gate_out = self.gate_proj(h)
        up_out = self.up_proj(h)
        if HAS_FUSED:
            x = x + self.down_proj(voxtral_kernels.fused_silu_mul(gate_out, up_out))
        else:
            x = x + self.down_proj(F.silu(gate_out) * up_out)
        return x


# ─── Full Model ──────────────────────────────────────────────────────────────

class VoxtralModel:
    def __init__(self, model_path, device='cuda', dtype=torch.float16, compile=False):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path

        with open(os.path.join(model_path, 'params.json')) as f:
            self.p = json.load(f)

        ea = self.p['multimodal']['whisper_model_args']['encoder_args']
        da = self.p['multimodal']['whisper_model_args']['downsample_args']
        self.ds_factor = da['downsample_factor']
        self.enc_h = ea['dim']
        self.n_layers = self.p['n_layers']
        self.h = self.p['dim']
        self.n_kv_heads = self.p.get('n_kv_heads', 8)
        self.head_dim = self.p.get('head_dim', 128)
        self.delay_ms = self.p['multimodal']['whisper_model_args'].get(
            'encoder_args', {}).get('audio_encoding_args', {}).get(
            'transcription_delay_ms', 480)

        # Frame rate from config
        self.frame_rate = ea.get('audio_encoding_args', {}).get('frame_rate', 12.5)
        # delay_tokens = delay_ms / ms_per_frame
        ms_per_frame = 1000.0 / self.frame_rate
        self.delay_tokens = int(self.delay_ms / ms_per_frame)
        self.n_left_pad = 32  # streaming_n_left_pad_tokens from audio config

        self._build()
        self._load()
        if compile:
            self._try_compile()

        self.cos, self.sin = make_freqs(self.p['head_dim'], 8192, device=device, dtype=dtype)

        # Time conditioning: sinusoidal embedding of delay token count
        self.t_cond = self._compute_time_embedding(
            float(self.delay_tokens), self.h
        ).to(device=device, dtype=dtype)

        # Pre-compute ada_rms_norm modulation for all decoder layers
        # (t_cond is constant, so ada output is constant per layer)
        for layer in self.layers:
            layer.precompute_ada(self.t_cond)

    def _build(self):
        """Build model skeleton on meta device (zero memory allocation)."""
        import gc
        ea = self.p['multimodal']['whisper_model_args']['encoder_args']
        with torch.device('meta'):
            self.encoder = Encoder(ea['n_layers'], ea['dim'], ea['hidden_dim'], ea['head_dim'])
            self.projector = Projector(ea['hidden_dim'], self.h)
            self.layers = nn.ModuleList([
                DecLayer(self.h, self.p.get('ada_rms_norm_t_cond_dim', 32))
                for _ in range(self.n_layers)
            ])
            self.embed = nn.Embedding(self.p['vocab_size'], self.h)
            self.norm = RMSNorm(self.h)
        gc.collect()

    def _try_compile(self):
        """Optionally compile decoder layers with torch.compile.

        Fuses pointwise ops (RMSNorm, SiLU, ada_rms_norm, residuals) between
        Marlin matmuls, reducing kernel launch overhead at batch-1 decode.
        Falls back gracefully if torch.compile isn't available on this platform.
        """
        try:
            compiled = 0
            for i in range(len(self.layers)):
                self.layers[i] = torch.compile(self.layers[i], mode="default")
                compiled += 1
            print(f"  torch.compile: {compiled} decoder layers compiled")
        except Exception as e:
            print(f"  torch.compile not available ({e}), using eager mode")

    def _pack_lm_head(self):
        """Quantize the embedding weight to INT4 and pack for Marlin LM head.

        The LM head (output projection) reads the full 768 MB fp16 embedding
        matrix on every token — 21ms at Jetson bandwidth. By quantizing to INT4
        and using Marlin, this drops to ~4ms (5.8x faster).

        Must be called before decoder layers are loaded (maximum free GPU memory).
        """
        import gc
        embed_w = self.embed.weight.data  # [vocab_size, dim] fp16
        out_f, in_f = embed_w.shape

        if in_f % 128 != 0 or out_f % 256 != 0:
            print(f"  LM head: dims {out_f}x{in_f} not Marlin-compatible, keeping fp16")
            self._lm_head_marlin = False
            return

        t0 = time.time()
        # Compute per-group scales (group_size=128 along input dim)
        n_groups = in_f // GROUP_SIZE
        w_grouped = embed_w.float().reshape(out_f, n_groups, GROUP_SIZE)
        scales = (w_grouped.abs().amax(dim=-1) / 7.0).clamp(min=1e-6).half()  # [out_f, n_groups]
        del w_grouped

        # Pack with Marlin (re-uses embed_w in-place for the linear)
        linear_tmp = nn.Linear(in_f, out_f, bias=False, dtype=torch.half, device=embed_w.device)
        linear_tmp.weight.data = embed_w  # share, no copy
        ml = _marlin.Layer(in_f, out_f, groupsize=GROUP_SIZE)
        ml.pack(linear_tmp, scales)
        del linear_tmp, scales
        gc.collect()
        torch.cuda.empty_cache()

        self._lm_B = ml.B.to(embed_w.device)
        self._lm_s = ml.s.to(embed_w.device)
        self._lm_ws = torch.zeros(out_f // 128 * 16, dtype=torch.int, device=embed_w.device)
        self._lm_out_f = out_f
        self._lm_head_marlin = True

        mb = (self._lm_B.nelement() * self._lm_B.element_size() +
              self._lm_s.nelement() * self._lm_s.element_size()) / 1024**2
        print(f"  Marlin LM head packed: {out_f}x{in_f} -> {mb:.0f} MB ({time.time()-t0:.1f}s)")

    def _lm_head(self, h):
        """Compute LM head logits. Uses Marlin INT4 if available, else fp16."""
        if hasattr(self, '_lm_head_marlin') and self._lm_head_marlin:
            h_flat = h.view(-1, h.shape[-1])
            out = torch.empty(h_flat.shape[0], self._lm_out_f, dtype=h.dtype, device=h.device)
            _marlin.mul(h_flat, self._lm_B, out, self._lm_s, self._lm_ws)
            return out.view(*h.shape[:-1], self._lm_out_f)
        return F.linear(h, self.embed.weight)

    def _dql(self, f, prefix, dev, unpermute=None):
        B = f.get_tensor(f'{prefix}.B').to(dev)
        s = f.get_tensor(f'{prefix}.s').to(dev)
        if not HAS_MARLIN:
            raise RuntimeError(
                "Marlin INT4 kernel required but not installed. "
                "Install with: pip install marlin")
        return PrepackedMarlinLinear(B, s)

    def _set(self, module, name, tensor):
        """Replace a meta parameter with a real CUDA tensor."""
        module._parameters[name] = nn.Parameter(tensor, requires_grad=False)

    @staticmethod
    def _compute_time_embedding(t_value: float, dim: int, theta: float = 10000.0) -> torch.Tensor:
        """Sinusoidal embedding of scalar t_value into dim-dimensional vector."""
        half_dim = dim // 2
        inv_freq = torch.exp(-math.log(theta) * torch.arange(half_dim).float() / half_dim)
        emb = t_value * inv_freq
        return torch.cat([emb.cos(), emb.sin()])  # [dim]

    @staticmethod
    def _evict_cache():
        """Force kernel to reclaim page cache on Jetson unified memory."""
        import ctypes, gc
        for sz in [4, 3, 2, 1]:
            try:
                buf = ctypes.create_string_buffer(sz * 1024 * 1024 * 1024)
                del buf
                gc.collect()
                return
            except (MemoryError, OSError):
                continue

    def _load_section(self, path, load_fn):
        """Open safetensors, run load_fn, close, evict cache."""
        import gc
        D = str(self.device)
        with safe_open(path, framework='pt', device=D) as f:
            load_fn(f)
        gc.collect()
        torch.cuda.empty_cache()
        self._evict_cache()

    def _load(self):
        import gc
        path = os.path.join(self.model_path, 'consolidated.safetensors')
        print(f"Loading {path}...")
        t0 = time.time()
        D, T = self.device, self.dtype
        ep = 'mm_streams_embeddings.embedding_module'
        enc_prefix = f'{ep}.whisper_encoder'
        ea = self.p['multimodal']['whisper_model_args']['encoder_args']

        # Section 1: Embeddings + output norm
        def load_embed(f):
            self._set(self.embed, 'weight', f.get_tensor(f'{ep}.tok_embeddings.weight').to(T))
            self._set(self.norm, 'weight', f.get_tensor('norm.weight').to(T))
            print(f"  Embeddings loaded")
        self._load_section(path, load_embed)

        # Section 2: Encoder convolutions
        def load_enc_conv(f):
            self._set(self.encoder.conv1, 'weight', f.get_tensor(f'{enc_prefix}.conv_layers.0.conv.weight').to(T))
            self._set(self.encoder.conv1, 'bias', f.get_tensor(f'{enc_prefix}.conv_layers.0.conv.bias').to(T))
            self._set(self.encoder.conv2, 'weight', f.get_tensor(f'{enc_prefix}.conv_layers.1.conv.weight').to(T))
            self._set(self.encoder.conv2, 'bias', f.get_tensor(f'{enc_prefix}.conv_layers.1.conv.bias').to(T))
            self._set(self.encoder.norm, 'weight', f.get_tensor(f'{enc_prefix}.transformer.norm.weight').to(T))
            print(f"  Encoder conv loaded")
        self._load_section(path, load_enc_conv)

        # Section 3: Encoder layers (in batches of 8 to limit mmap cache growth)
        n_enc = ea['n_layers']
        batch = 8
        for b_start in range(0, n_enc, batch):
            b_end = min(b_start + batch, n_enc)
            def load_enc_batch(f, start=b_start, end=b_end):
                for i in range(start, end):
                    lp = f'{enc_prefix}.transformer.layers.{i}'
                    el = self.encoder.layers[i]
                    self._set(el.attn.wq, 'weight', f.get_tensor(f'{lp}.attention.wq.weight').to(T))
                    self._set(el.attn.wq, 'bias', f.get_tensor(f'{lp}.attention.wq.bias').to(T))
                    self._set(el.attn.wk, 'weight', f.get_tensor(f'{lp}.attention.wk.weight').to(T))
                    self._set(el.attn.wv, 'weight', f.get_tensor(f'{lp}.attention.wv.weight').to(T))
                    self._set(el.attn.wv, 'bias', f.get_tensor(f'{lp}.attention.wv.bias').to(T))
                    self._set(el.attn.wo, 'weight', f.get_tensor(f'{lp}.attention.wo.weight').to(T))
                    self._set(el.attn.wo, 'bias', f.get_tensor(f'{lp}.attention.wo.bias').to(T))
                    self._set(el.an, 'weight', f.get_tensor(f'{lp}.attention_norm.weight').to(T))
                    self._set(el.fn, 'weight', f.get_tensor(f'{lp}.ffn_norm.weight').to(T))
                    self._set(el.w1, 'weight', f.get_tensor(f'{lp}.feed_forward.w1.weight').to(T))
                    self._set(el.w2, 'weight', f.get_tensor(f'{lp}.feed_forward.w2.weight').to(T))
                    self._set(el.w2, 'bias', f.get_tensor(f'{lp}.feed_forward.w2.bias').to(T))
                    self._set(el.w3, 'weight', f.get_tensor(f'{lp}.feed_forward.w3.weight').to(T))
                print(f"  Encoder layers {start}-{end-1} loaded")
            self._load_section(path, load_enc_batch)

        print(f"  Encoder loaded ({n_enc} layers)")

        # Section 4: Projector
        def load_proj(f):
            pp = f'{ep}.audio_language_projection'
            self._set(self.projector.l1, 'weight', f.get_tensor(f'{pp}.0.weight').to(T))
            self._set(self.projector.l2, 'weight', f.get_tensor(f'{pp}.2.weight').to(T))
            print(f"  Projector loaded")
        self._load_section(path, load_proj)

        # Section 4b: Pack Marlin INT4 LM head (before decoder layers, max free mem)
        if HAS_MARLIN:
            self._pack_lm_head()

        # Section 5: LM decoder layers (in batches of 13)
        dec_batch = 13
        for b_start in range(0, self.n_layers, dec_batch):
            b_end = min(b_start + dec_batch, self.n_layers)
            def load_dec_batch(f, start=b_start, end=b_end):
                for i in range(start, end):
                    lp = f'layers.{i}'
                    dl = self.layers[i]
                    self._set(dl.an, 'weight', f.get_tensor(f'{lp}.attention_norm.weight').to(T))
                    self._set(dl.fn, 'weight', f.get_tensor(f'{lp}.ffn_norm.weight').to(T))
                    self._set(dl.ada0, 'weight', f.get_tensor(f'{lp}.ada_rms_norm_t_cond.0.weight').to(T))
                    self._set(dl.ada2, 'weight', f.get_tensor(f'{lp}.ada_rms_norm_t_cond.2.weight').to(T))
                    dl.attn.q_proj = self._dql(f, f'{lp}.self_attn.q_proj', D)
                    dl.attn.k_proj = self._dql(f, f'{lp}.self_attn.k_proj', D)
                    dl.attn.v_proj = self._dql(f, f'{lp}.self_attn.v_proj', D)
                    dl.attn.o_proj = self._dql(f, f'{lp}.self_attn.o_proj', D)
                    dl.gate_proj = self._dql(f, f'{lp}.mlp.gate_proj', D)
                    dl.up_proj = self._dql(f, f'{lp}.mlp.up_proj', D)
                    dl.down_proj = self._dql(f, f'{lp}.mlp.down_proj', D)
                print(f"  LM layers {start}-{end-1} loaded")
            self._load_section(path, load_dec_batch)

        print(f"  LM decoder loaded ({self.n_layers} layers, Marlin fused INT4)")
        gc.collect()
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"  Done in {time.time()-t0:.1f}s, GPU: {mem:.2f} GB")

    def _load_tokenizer(self):
        """Load tekken tokenizer for decoding."""
        try:
            from mistral_common.tokens.tokenizers.tekken import Tekkenizer
            self.tokenizer = Tekkenizer.from_file(
                os.path.join(self.model_path, 'tekken.json'))
            print("  Tekken tokenizer loaded")
        except ImportError:
            try:
                from mistral_common.tokens.tokenizers.tekken import Tekken
                self.tokenizer = Tekken.from_file(
                    os.path.join(self.model_path, 'tekken.json'))
                print("  Tekken tokenizer loaded (legacy)")
            except ImportError:
                self.tokenizer = None
                print("  WARNING: mistral_common not available, using fallback decoder")

    def decode_tokens(self, ids):
        if self.tokenizer is not None:
            try:
                return self.tokenizer.decode(ids)
            except:
                pass
        # Fallback: decode as UTF-8 byte tokens
        # Token IDs 0-255 are byte tokens in tekken
        result = bytearray()
        for tid in ids:
            if 0 <= tid <= 255:
                result.append(tid)
            elif 256 <= tid < 131072:
                # BPE merge token — would need full vocab to decode
                result.extend(f'[{tid}]'.encode())
        try:
            return result.decode('utf-8', errors='replace')
        except:
            return str(ids)

    @staticmethod
    def _pad_audio(audio: np.ndarray) -> np.ndarray:
        """Pad audio for streaming alignment: left silence + alignment + right silence.

        Left padding of N_LEFT_PAD_TOKENS tokens aligns with the SPAD tokens
        in the prompt. Right padding accounts for delay + offline buffer.
        """
        n = len(audio)
        left_pad = N_LEFT_PAD_TOKENS * RAW_AUDIO_PER_TOK
        right_align = (RAW_AUDIO_PER_TOK - (n % RAW_AUDIO_PER_TOK)) % RAW_AUDIO_PER_TOK
        right_pad = right_align + N_RIGHT_PAD_BASE * RAW_AUDIO_PER_TOK
        return np.pad(audio.astype(np.float32), (left_pad, right_pad))

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, max_tokens: int = 512) -> str:
        """Transcribe audio (float32, 16kHz mono) to text."""
        t0 = time.time()

        # Evict page cache before inference (Jetson unified memory)
        self._evict_cache()
        free, _ = torch.cuda.mem_get_info(0)
        print(f"  CUDA free before inference: {free/1024**3:.2f} GB")

        # 0. Pad audio for streaming alignment
        audio = self._pad_audio(audio)
        print(f"  padded audio: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.1f}s)")

        # 1. Mel spectrogram
        mel = self._mel(audio)
        # Pad mel for downsample alignment
        enc_out_len = (mel.shape[2] - 1) // 2 + 1  # after conv stride-2
        remainder = enc_out_len % self.ds_factor
        if remainder != 0:
            mel = F.pad(mel, (0, (self.ds_factor - remainder) * 2))
        print(f"  mel: {mel.shape}")

        # 2. Encode
        t1 = time.time()
        enc = self.encoder(mel)  # [1, T_enc, 1280]
        print(f"  enc: {enc.shape} ({time.time()-t1:.2f}s)")
        del mel  # free mel to save memory

        # 3. Downsample 4x: concat groups of ds_factor frames
        T_enc = enc.shape[1]
        T_ds = T_enc // self.ds_factor
        enc_ds = enc[:, :T_ds * self.ds_factor, :].reshape(
            1, T_ds, self.ds_factor * self.enc_h)  # [1, T_ds, 5120]
        del enc  # free encoder output

        # 4. Project (adapter)
        adapter = self.projector(enc_ds)  # [1, T_ds, 3072]
        del enc_ds
        print(f"  adapter: {adapter.shape}")

        # 5. Build prompt: [BOS] + [SPAD] * (n_left_pad + delay_tokens)
        prompt_len = 1 + self.n_left_pad + self.delay_tokens
        prompt_ids = [TOKEN_BOS] + [TOKEN_STREAMING_PAD] * (self.n_left_pad + self.delay_tokens)

        # Clamp prompt to adapter length
        if prompt_len > T_ds:
            prompt_len = T_ds
            prompt_ids = prompt_ids[:T_ds]

        print(f"  prompt: {prompt_len} tokens, adapter: {T_ds} positions, gen budget: {T_ds - prompt_len}")

        # 7. Allocate KV cache for the full sequence
        kv_cache = KVCache(self.n_layers, T_ds, self.n_kv_heads, self.head_dim,
                           self.device, self.dtype)

        # 8. Prefill: build embeddings = adapter + tok_embed for prompt positions
        prompt_tok = torch.tensor([prompt_ids], device=self.device)
        tok_emb = self.embed(prompt_tok)  # [1, prompt_len, 3072]
        input_emb = adapter[:, :prompt_len, :] + tok_emb  # ADDITION

        cos = self.cos[:prompt_len]
        sin = self.sin[:prompt_len]

        h = input_emb
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, cache=kv_cache, layer_idx=i,
                      is_causal=True, t_cond=self.t_cond)
        kv_cache.advance(prompt_len)

        h = self.norm(h)

        # lm_head (Marlin INT4 if available, else fp16 tied embed)
        logits = self._lm_head(h[:, -1:, :])
        next_tok = logits.argmax(-1).item()
        # Diagnostic: top-5 tokens and logit stats
        topk = torch.topk(logits[0, 0], 10)
        print(f"  prefill done ({time.time()-t1:.2f}s), first token: {next_tok}")
        print(f"  logits: min={logits.min():.2f} max={logits.max():.2f} mean={logits.mean():.4f}")
        print(f"  top-10: {list(zip(topk.indices.tolist(), [f'{v:.2f}' for v in topk.values.tolist()]))}")
        print(f"  adapter stats: min={adapter.min():.4f} max={adapter.max():.4f} mean={adapter.mean():.6f}")

        # 9. Generate: continue from prompt_len to T_ds
        generated = []
        pos = prompt_len
        t2 = time.time()

        # Pre-allocate token tensor to avoid per-step allocation
        _tok_buf = torch.empty(1, 1, dtype=torch.long, device=self.device)

        while pos < T_ds and next_tok != TOKEN_EOS and len(generated) < max_tokens:
            generated.append(next_tok)

            # Input = adapter[pos] + tok_embed(next_tok)
            _tok_buf.fill_(next_tok)
            te = self.embed(_tok_buf)
            inp = adapter[:, pos:pos+1, :] + te

            cos_s = self.cos[pos:pos+1]
            sin_s = self.sin[pos:pos+1]

            h = inp
            for i, layer in enumerate(self.layers):
                h = layer(h, cos_s, sin_s, cache=kv_cache, layer_idx=i,
                          is_causal=False, t_cond=self.t_cond)
            kv_cache.advance(1)

            h = self.norm(h)
            logits = self._lm_head(h)
            next_tok = logits[:, -1, :].argmax(-1).item()
            pos += 1

            # Progress every 25 tokens
            if len(generated) % 25 == 0:
                elapsed = time.time() - t2
                tps = len(generated) / max(elapsed, 0.001)
                n_text = sum(1 for t in generated if t >= 1000)
                print(f"    step {pos}/{T_ds}: {len(generated)} tok ({n_text} text), {tps:.1f} tok/s")

        if next_tok == TOKEN_EOS:
            generated.append(TOKEN_EOS)

        dt = time.time() - t2
        n_gen = len(generated)

        # Filter: text tokens are >= 1000 (special tokens are < 1000)
        text_toks = [t for t in generated if t >= 1000]
        special_toks = [t for t in generated if t < 1000]
        text = self.decode_tokens(text_toks)

        print(f"  gen: {n_gen} tokens ({len(special_toks)} special, {len(text_toks)} text) in {dt:.2f}s "
              f"({n_gen/max(dt,0.001):.1f} tok/s)")
        if special_toks:
            print(f"  special tokens: {sorted(set(special_toks))[:10]}")
        if text_toks:
            print(f"  first text IDs: {text_toks[:20]}")
        print(f"  total: {time.time()-t0:.2f}s")

        return text

    def _mel(self, audio: np.ndarray) -> torch.Tensor:
        """Compute Whisper-style log-mel spectrogram.

        Uses log10, max-relative normalization, and [0,1] scaling matching
        the standard Whisper preprocessing pipeline.
        """
        at = torch.from_numpy(audio).float()
        win = torch.hann_window(WINDOW_SIZE)
        stft = torch.stft(at, WINDOW_SIZE, HOP_LENGTH, WINDOW_SIZE, win,
                         return_complex=True)
        magnitudes = stft[..., :-1].abs().pow(2)

        # Mel filterbank
        fb = self._melfb(WINDOW_SIZE // 2 + 1, N_MELS, SAMPLE_RATE, 0.0, 8000.0)
        mel = fb @ magnitudes

        # Voxtral log normalization: fixed global max, NOT per-sample
        log_mel = torch.log10(mel.clamp(min=1e-10))
        log_mel = torch.clamp(log_mel, min=GLOBAL_LOG_MEL_MAX - 8.0)  # floor at -6.5
        log_mel = (log_mel + 4.0) / 4.0

        return log_mel.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _melfb(n_fft_bins, n_mels, sr, fmin=0.0, fmax=8000.0):
        """Slaney mel filterbank matching transformers.audio_utils.mel_filter_bank."""
        # Slaney mel scale (piecewise linear/log, NOT HTK)
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp  # 15.0
        logstep = np.log(6.4) / 27.0

        def hz_to_mel(f):
            f = np.asarray(f, dtype=np.float64)
            mel = np.where(f < min_log_hz, f / f_sp,
                          min_log_mel + np.log(f / min_log_hz) / logstep)
            return mel

        def mel_to_hz(m):
            m = np.asarray(m, dtype=np.float64)
            hz = np.where(m < min_log_mel, m * f_sp,
                         min_log_hz * np.exp(logstep * (m - min_log_mel)))
            return hz

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mels = np.linspace(mel_min, mel_max, n_mels + 2)
        freqs = mel_to_hz(mels)

        fft_freqs = np.linspace(0, sr / 2, n_fft_bins)
        fb = np.zeros((n_mels, n_fft_bins))
        for i in range(n_mels):
            low, center, high = freqs[i], freqs[i + 1], freqs[i + 2]
            for j in range(n_fft_bins):
                if low <= fft_freqs[j] <= center and center > low:
                    fb[i, j] = (fft_freqs[j] - low) / (center - low)
                elif center < fft_freqs[j] <= high and high > center:
                    fb[i, j] = (high - fft_freqs[j]) / (high - center)

        # Slaney normalization: area = 1 per filter
        enorm = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])
        fb *= enorm[:, np.newaxis]

        return torch.tensor(fb, dtype=torch.float32)


# ─── WebSocket Server ────────────────────────────────────────────────────────

async def handle_ws(ws, model):
    sid = str(uuid.uuid4())[:8]
    await ws.send(json.dumps({"type": "session.created", "session": {"id": sid}}))

    buf = bytearray()
    async for msg in ws:
        try:
            ev = json.loads(msg)
            t = ev.get("type", "")

            if t == "session.update":
                pass  # model name ignored, we only have one

            elif t == "input_audio_buffer.append":
                buf.extend(base64.b64decode(ev.get("audio", "")))

            elif t == "input_audio_buffer.commit":
                if ev.get("final"):
                    break
                if not buf:
                    continue

                pcm = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
                buf = bytearray()
                dur = len(pcm) / SAMPLE_RATE
                print(f"[{sid}] {dur:.1f}s audio")

                text = model.transcribe(pcm)

                if text:
                    await ws.send(json.dumps({"type": "transcription.delta", "delta": text}))
                await ws.send(json.dumps({
                    "type": "transcription.done", "text": text,
                    "usage": {"audio_duration_s": round(dur, 1)}
                }))
                print(f"[{sid}] -> {text[:100]}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            await ws.send(json.dumps({"type": "error", "error": str(e)}))


async def serve(model, host='0.0.0.0', port=8000):
    import websockets
    print(f"\nWebSocket server at ws://{host}:{port}/v1/realtime")

    async def handler(ws, path=None):
        await handle_ws(ws, model)

    try:
        async with websockets.serve(handler, host, port):
            print("Ready.")
            await asyncio.Future()
    except TypeError:
        await websockets.serve(handler, host, port)
        print("Ready.")
        await asyncio.Future()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='/workspace/voxtral-quant/models/voxtral-rtn-4bit-packed-vllm')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--test', help='WAV file to transcribe (skip server)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    args = ap.parse_args()

    print("=" * 60)
    print("Voxtral Mini 4B Realtime — Jetson Orin Nano 8GB")
    print("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory/1024**3:.1f} GB")
        free, total = torch.cuda.mem_get_info(0)
        print(f"CUDA free: {free/1024**3:.2f} GB / {total/1024**3:.2f} GB")
        # Force kernel to reclaim page cache (critical on Jetson unified memory)
        if free < 5 * 1024**3:
            import ctypes, gc
            print("Reclaiming page cache...")
            for sz in [4, 3, 2, 1]:
                try:
                    buf = ctypes.create_string_buffer(sz * 1024 * 1024 * 1024)
                    del buf
                    gc.collect()
                    break
                except (MemoryError, OSError):
                    continue
            free2, _ = torch.cuda.mem_get_info(0)
            print(f"CUDA free after reclaim: {free2/1024**3:.2f} GB")

    model = VoxtralModel(args.model_path, args.device, compile=not args.no_compile)
    model._load_tokenizer()

    if args.test:
        import soundfile as sf
        audio, sr = sf.read(args.test, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(1)
        if sr != SAMPLE_RATE:
            try:
                import soxr
                audio = soxr.resample(audio, sr, SAMPLE_RATE, quality='HQ')
            except ImportError:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        print(f"\nTest: {args.test} ({len(audio)/SAMPLE_RATE:.1f}s)")
        text = model.transcribe(audio)
        print(f"\nResult: {text}")
    else:
        asyncio.run(serve(model, port=args.port))


if __name__ == '__main__':
    main()
