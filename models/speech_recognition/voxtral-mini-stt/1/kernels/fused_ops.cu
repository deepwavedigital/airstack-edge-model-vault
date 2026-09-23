// Copyright 2025 Mistral AI
// SPDX-License-Identifier: Apache-2.0
//
// Modified by Deepwave Digital, Inc. in 2026.
//
// Fused CUDA kernels for Voxtral decoder on Jetson Orin Nano (SM87).
//
// Three kernels that collapse ~500 PyTorch kernel launches per token into ~80:
//   1. fused_rmsnorm:  5-6 kernels → 1 per call (52 calls/token)
//   2. fused_rope_hf:  ~14 kernels → 2 per call (26 calls/token)
//   3. fused_silu_mul: 2 kernels → 1 per call (26 calls/token)
//
// Build: JIT via torch.utils.cpp_extension.load() with -arch=sm_87
// All kernels: fp16 I/O, fp32 internal accumulation where needed.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================================
// Warp-level reduction
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// Kernel 1: Fused RMSNorm
// ============================================================================
//
// Replaces: x.float() → pow(2) → mean → add(eps) → rsqrt → mul(x) → half → mul(weight)
// One block per row. 256 threads, each handles dim/256 elements.
// Warp shuffle + shared memory for cross-warp reduction.

__global__ void fused_rmsnorm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ out,
    int dim,
    float eps
) {
    const int row = blockIdx.x;
    const half* x_row = x + (int64_t)row * dim;
    half* out_row = out + (int64_t)row * dim;

    // Phase 1: partial sum of squares (fp32 accumulation)
    float partial = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float v = __half2float(x_row[i]);
        partial += v * v;
    }

    // Phase 2: warp reduction
    partial = warp_reduce_sum(partial);

    // Phase 3: cross-warp reduction via shared memory
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0)
        warp_sums[warp_id] = partial;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane] : 0.0f;
        total = warp_reduce_sum(total);
    }

    // Phase 4: broadcast norm factor
    __shared__ float s_norm;
    if (threadIdx.x == 0)
        s_norm = rsqrtf(total / (float)dim + eps);
    __syncthreads();

    // Phase 5: normalize and write
    const float nf = s_norm;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        float v = __half2float(x_row[i]);
        float wt = __half2float(w[i]);
        out_row[i] = __float2half(v * nf * wt);
    }
}

torch::Tensor fused_rmsnorm(torch::Tensor x, torch::Tensor weight, float eps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf);
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kHalf);

    auto x_c = x.contiguous();
    auto out = torch::empty_like(x_c);
    const int dim = x_c.size(-1);
    const int rows = x_c.numel() / dim;

    fused_rmsnorm_kernel<<<rows, BLOCK_SIZE>>>(
        reinterpret_cast<const half*>(x_c.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        dim, eps
    );

    return out;
}

// ============================================================================
// Kernel 2: Fused Half-Rotation RoPE (in-place on contiguous copy)
// ============================================================================
//
// Half-rotation format: q[..., :hd/2] = real, q[..., hd/2:] = imaginary
// Replaces: float() cast + 4 slices + 4 muls + 2 subs + 2 cats + half() cast
//
// Works for both decode (T=1) and prefill (T>1).
// Data layout after .contiguous(): q[h, t, d] = q_ptr[h*T*hd + t*hd + d]

__global__ void fused_rope_kernel(
    half* __restrict__ data,         // [n_heads, T, head_dim] contiguous
    const half* __restrict__ cos_ptr, // [T, half_dim]
    const half* __restrict__ sin_ptr, // [T, half_dim]
    int n_heads,
    int seq_len,
    int head_dim
) {
    const int half_dim = head_dim >> 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_heads * seq_len * half_dim;
    if (idx >= total) return;

    const int i = idx % half_dim;
    const int t = (idx / half_dim) % seq_len;
    const int h = idx / (half_dim * seq_len);

    const int base = h * seq_len * head_dim + t * head_dim;
    const int cs = t * half_dim + i;

    const float c = __half2float(cos_ptr[cs]);
    const float s = __half2float(sin_ptr[cs]);
    const float real = __half2float(data[base + i]);
    const float imag = __half2float(data[base + half_dim + i]);

    data[base + i]            = __float2half(real * c - imag * s);
    data[base + half_dim + i] = __float2half(real * s + imag * c);
}

std::vector<torch::Tensor> fused_rope_hf(
    torch::Tensor q,   // [B, n_q_heads, T, head_dim]
    torch::Tensor k,   // [B, n_kv_heads, T, head_dim]
    torch::Tensor cos,  // [T, head_dim/2]
    torch::Tensor sin   // [T, head_dim/2]
) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kHalf);

    // Make contiguous copies (needed because transpose makes q/k non-contiguous)
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto cos_c = cos.contiguous();
    auto sin_c = sin.contiguous();

    const int B = q_c.size(0);
    const int n_q = q_c.size(1);
    const int T = q_c.size(2);
    const int hd = q_c.size(3);
    const int n_kv = k_c.size(1);
    const int half_dim = hd >> 1;

    // Process each batch item (B=1 in practice, so no overhead)
    for (int b = 0; b < B; b++) {
        half* q_ptr = reinterpret_cast<half*>(q_c.data_ptr<at::Half>()) + b * n_q * T * hd;
        half* k_ptr = reinterpret_cast<half*>(k_c.data_ptr<at::Half>()) + b * n_kv * T * hd;

        const int total_q = n_q * T * half_dim;
        const int total_kv = n_kv * T * half_dim;

        fused_rope_kernel<<<(total_q + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            q_ptr,
            reinterpret_cast<const half*>(cos_c.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(sin_c.data_ptr<at::Half>()),
            n_q, T, hd
        );

        fused_rope_kernel<<<(total_kv + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            k_ptr,
            reinterpret_cast<const half*>(cos_c.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(sin_c.data_ptr<at::Half>()),
            n_kv, T, hd
        );
    }

    return {q_c, k_c};
}

// ============================================================================
// Kernel 3: Fused SiLU * Multiply
// ============================================================================
//
// Replaces: F.silu(gate) * up  (2 separate kernels)
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

__global__ void fused_silu_mul_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float g = __half2float(gate[idx]);
        const float u = __half2float(up[idx]);
        out[idx] = __float2half((g / (1.0f + expf(-g))) * u);
    }
}

torch::Tensor fused_silu_mul(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == torch::kHalf);
    TORCH_CHECK(up.is_cuda() && up.scalar_type() == torch::kHalf);

    auto gate_c = gate.contiguous();
    auto up_c = up.contiguous();
    auto out = torch::empty_like(gate_c);
    const int n = gate_c.numel();

    fused_silu_mul_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<const half*>(gate_c.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(up_c.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        n
    );

    return out;
}

// ============================================================================
// Module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rmsnorm", &fused_rmsnorm, "Fused RMSNorm (fp16 I/O, fp32 accumulation)");
    m.def("fused_rope_hf", &fused_rope_hf, "Fused half-rotation RoPE (fp16, in-place on contiguous copy)");
    m.def("fused_silu_mul", &fused_silu_mul, "Fused SiLU * mul (fp16)");
}
