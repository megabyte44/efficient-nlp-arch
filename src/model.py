"""Minimal decoder-only Transformer.

`attn_type` is a deliberate seam: Stage 2 of the roadmap swaps this block
for sparse/linear attention or a selective-state block, and everything
else (embeddings, MLP, training loop, benchmark harness) stays fixed so
comparisons are apples-to-apples.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def chunked_causal_attention(q, k, v, chunk_size, dropout_p, window_size=None):
    """Blocked, online-softmax causal attention (the FlashAttention tiling
    trick): queries and keys are processed in fixed-size blocks and the
    softmax is accumulated incrementally (running max/sum/weighted-value),
    so the full TxT score matrix is never materialized. Mathematically
    identical to plain (or windowed) causal softmax attention -- this is
    exact attention, not an approximation.

    Causal key-blocks that are provably irrelevant to a query block (in the
    future, or -- when `window_size` is given -- entirely outside the
    trailing window) are skipped outright: real compute savings, not just
    memory savings, closing the gap `LocalCausalSelfAttention`'s plain mask
    implementation leaves open.
    """
    B, H, T, Dh = q.shape
    scale = 1.0 / math.sqrt(Dh)
    neg_inf = -1e9
    n_chunks = math.ceil(T / chunk_size)
    out = torch.empty_like(q)

    for qi in range(n_chunks):
        q_start, q_end = qi * chunk_size, min((qi + 1) * chunk_size, T)
        q_blk = q[:, :, q_start:q_end]
        qlen = q_end - q_start

        running_max = torch.full((B, H, qlen, 1), neg_inf, device=q.device, dtype=q.dtype)
        running_sum = torch.zeros(B, H, qlen, 1, device=q.device, dtype=q.dtype)
        acc = torch.zeros(B, H, qlen, Dh, device=q.device, dtype=q.dtype)

        min_kj = 0
        if window_size is not None:
            window_start = max(0, q_start - window_size + 1)
            min_kj = window_start // chunk_size

        for kj in range(min_kj, qi + 1):
            k_start, k_end = kj * chunk_size, min((kj + 1) * chunk_size, T)
            k_blk = k[:, :, k_start:k_end]
            v_blk = v[:, :, k_start:k_end]

            scores = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * scale

            qi_idx = torch.arange(q_start, q_end, device=q.device).unsqueeze(-1)
            kj_idx = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
            allowed = qi_idx >= kj_idx
            if window_size is not None:
                allowed = allowed & ((qi_idx - kj_idx) < window_size)
            scores = scores.masked_fill(~allowed, neg_inf)

            block_max = scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, block_max)
            correction = torch.exp(running_max - new_max)
            exp_scores = torch.exp(scores - new_max)
            if dropout_p > 0:
                exp_scores = F.dropout(exp_scores, p=dropout_p, training=True)

            running_sum = running_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
            acc = acc * correction + torch.matmul(exp_scores, v_blk)
            running_max = new_max

        out[:, :, q_start:q_end] = acc / running_sum
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.chunked = cfg.get("chunked", False)
        self.chunk_size = cfg.get("chunk_size", 32)
        self.qkv = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=cfg["bias"])
        self.proj = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])
        self.attn_dropout = nn.Dropout(cfg["dropout"])
        self.resid_dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        if self.chunked:
            y = chunked_causal_attention(q, k, v, self.chunk_size, dropout_p)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class LocalCausalSelfAttention(nn.Module):
    """Causal attention restricted to a fixed-size trailing window.

    Query at position i may attend to keys in [i - window_size + 1, i]
    (still causal — no peeking ahead). Reduces the attention term from
    O(T) per token to O(window_size) per token, independent of sequence
    length once T > window_size. Implemented as an explicit boolean mask
    over full QK^T rather than a blocked/sparse kernel — correct and easy
    to verify, but on CPU this doesn't skip the masked-out multiplications,
    so wall-clock latency won't show the saving that analytical FLOPs do.
    A real speedup needs a chunked/blocked implementation (candidate
    follow-up, see ROADMAP.md).
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.window_size = cfg["window_size"]
        self.chunked = cfg.get("chunked", False)
        self.chunk_size = cfg.get("chunk_size", 32)
        self.qkv = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=cfg["bias"])
        self.proj = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])
        self.attn_dropout = nn.Dropout(cfg["dropout"])
        self.resid_dropout = nn.Dropout(cfg["dropout"])

        block_size = cfg["block_size"]
        pos = torch.arange(block_size)
        distance = pos[:, None] - pos[None, :]  # i - j
        allowed = (distance >= 0) & (distance < self.window_size)  # causal + within window
        self.register_buffer("attn_mask", allowed, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        if self.chunked:
            y = chunked_causal_attention(q, k, v, self.chunk_size, dropout_p, window_size=self.window_size)
        else:
            mask = self.attn_mask[:T, :T]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class DilatedCausalSelfAttention(nn.Module):
    """Causal attention over fixed dilated segments (LongNet's core primitive).

    The sequence is cut into non-overlapping segments of `segment_length`.
    Within a segment, position i only attends to positions j with the same
    residue mod `dilation_rate` (still causal — no peeking ahead). This is
    a fixed sparse pattern: O(segment_length / dilation_rate) keys per query
    instead of O(T), independent of how many segments there are.

    This implements the single (segment_length, dilation_rate) primitive,
    not LongNet's further refinement of mixing several such settings across
    heads — same incremental scope `local` used for plain windowing.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.segment_length = cfg["segment_length"]
        self.dilation_rate = cfg["dilation_rate"]
        self.qkv = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=cfg["bias"])
        self.proj = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])
        self.attn_dropout = nn.Dropout(cfg["dropout"])
        self.resid_dropout = nn.Dropout(cfg["dropout"])

        block_size = cfg["block_size"]
        pos = torch.arange(block_size)
        same_segment = (pos[:, None] // self.segment_length) == (pos[None, :] // self.segment_length)
        same_residue = (pos[:, None] % self.dilation_rate) == (pos[None, :] % self.dilation_rate)
        causal = pos[:, None] >= pos[None, :]
        allowed = same_segment & same_residue & causal
        self.register_buffer("attn_mask", allowed, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        mask = self.attn_mask[:T, :T]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class S4DMixer(nn.Module):
    """Diagonal state-space sequence mixer (S4D), computed via its closed-form
    convolutional view instead of a sequential recurrence.

    Each of the n_embd channels holds an independent linear SSM with a
    complex diagonal state matrix A (d_state internal dims per channel).
    Because A is diagonal, the causal impulse response
    K[t] = 2*Re(C * exp(t*dt*A) * B_bar) has a closed form, and
    y = causal_conv(u, K) is computed in one shot via FFT in O(T log T) --
    no sequential scan, and no O(T^2) attention term at all. This is the
    core "S4 trick": an exact linear recurrence, evaluated in parallel over
    the whole sequence at once.

    No qkv/heads -- this replaces the attention sub-layer entirely (the
    "selective-state block as an alternative to attention" the roadmap
    names for Stage 2), not a variant of attention itself.
    """

    def __init__(self, cfg):
        super().__init__()
        n_embd = cfg["n_embd"]
        d_state = cfg["d_state"]
        self.n_embd = n_embd
        self.d_state = d_state

        dt_min, dt_max = 0.001, 0.1
        log_dt = torch.rand(n_embd) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # S4D-Lin init: A_n = -1/2 + i*pi*n, shared at init, learned per channel.
        log_A_real = torch.log(0.5 * torch.ones(n_embd, d_state))
        A_imag = math.pi * torch.arange(d_state).float().expand(n_embd, d_state).clone()
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        self.B = nn.Parameter(torch.ones(n_embd, d_state, dtype=torch.cfloat))
        C_real = torch.randn(n_embd, d_state) / math.sqrt(d_state)
        C_imag = torch.randn(n_embd, d_state) / math.sqrt(d_state)
        self.C = nn.Parameter(torch.complex(C_real, C_imag))
        self.D = nn.Parameter(torch.ones(n_embd))

        self.proj = nn.Linear(n_embd, n_embd, bias=cfg["bias"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def _kernel(self, T, device):
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (n_embd, d_state)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)  # (n_embd, d_state)
        B_bar = (A_bar - 1) / A * self.B  # (n_embd, d_state)

        t = torch.arange(T, device=device, dtype=A_bar.real.dtype)
        A_bar_pow = A_bar.unsqueeze(-1) ** t  # (n_embd, d_state, T)
        kernel = torch.einsum("cn,cnt->ct", self.C * B_bar, A_bar_pow)
        return 2 * kernel.real  # (n_embd, T)

    def forward(self, x):
        _, T, _ = x.shape
        u = x.transpose(1, 2)  # (B, n_embd, T)
        K = self._kernel(T, x.device)  # (n_embd, T)

        u_f = torch.fft.rfft(u, n=2 * T)
        K_f = torch.fft.rfft(K, n=2 * T)
        y = torch.fft.irfft(u_f * K_f.unsqueeze(0), n=2 * T)[..., :T]
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * u

        y = y.transpose(1, 2)  # (B, T, n_embd)
        return self.dropout(self.proj(y))


def mamba2_sequential_scan(delta, A, B_seq, x_in, C_seq, D, n_heads):
    """Reference sequential scan for the Mamba-2/SSD-style recurrence, where
    `A` is a single scalar per *head* -- shared across every channel within
    that head (`headdim = d_inner / n_heads`) and across every state dim --
    unlike `MambaMixer`'s Mamba-1/S6 formulation, where `A` is a full
    (d_inner, d_state) diagonal (independent decay per channel *and* per
    state). This is Phase 4 of the Mamba-scan investigation (see
    ROADMAP.md): a distinct architecture, not a modification of the
    existing MambaMixer, checking what restricting `A` this way buys (or
    costs) rather than trying to make the existing model imitate it.

    Shapes: delta (B,T,n_heads), A (n_heads,), B_seq/C_seq (B,T,d_state),
    x_in (B,T,d_inner) with d_inner = n_heads * headdim, D (d_inner,).
    Returns (B,T,d_inner).
    """
    Bsz, T, d_inner = x_in.shape
    headdim = d_inner // n_heads
    d_state = B_seq.shape[-1]
    x_in_h = x_in.view(Bsz, T, n_heads, headdim)

    state = x_in.new_zeros(Bsz, n_heads, headdim, d_state)
    ys = []
    for t in range(T):
        delta_t = delta[:, t]  # (B, H)
        A_bar = torch.exp(delta_t * A)  # (B, H) -- one scalar per head, not per (channel, state)
        deltaB_x = (
            delta_t.view(Bsz, n_heads, 1, 1)
            * B_seq[:, t].view(Bsz, 1, 1, d_state)
            * x_in_h[:, t].unsqueeze(-1)
        )  # (B, H, P, N)
        state = A_bar.view(Bsz, n_heads, 1, 1) * state + deltaB_x
        y_t = torch.einsum("bhpn,bn->bhp", state, C_seq[:, t])
        ys.append(y_t)
    y = torch.stack(ys, dim=1)  # (B, T, H, P)
    return y.reshape(Bsz, T, d_inner) + D * x_in


def mamba2_chunked_scan(delta, A, B_seq, x_in, C_seq, D, n_heads, chunk_size):
    """Chunked scan for the same Mamba-2/SSD-style recurrence, exploiting
    exactly the restriction `mamba2_sequential_scan` describes: because the
    decay depends only on (batch, time, head) -- not on channel-within-head
    or state dim -- the intra-chunk decay tensor built below is shaped
    (B, chunk_size, chunk_size, n_heads), with no (headdim, d_state) factor
    at all. Compare to Phase 1/2's `mamba_chunked_scan` (a different
    branch/experiment), whose decay tensor was
    (B, chunk_size, chunk_size, d_inner, d_state) -- exactly
    `headdim * d_state` times larger per chunk -- and which Phase 2 found
    caused a severe, hardware-sensitive slowdown at larger chunk sizes or
    model widths. This function exists to check whether this restriction
    actually removes that blowup, at the cost of the coarser (per-head
    rather than per-channel-and-state) selectivity described above.
    """
    Bsz, T, d_inner = x_in.shape
    headdim = d_inner // n_heads
    d_state = B_seq.shape[-1]
    device = delta.device
    x_in_h = x_in.view(Bsz, T, n_heads, headdim)

    y = x_in.new_empty(Bsz, T, n_heads, headdim)
    state = x_in.new_zeros(Bsz, n_heads, headdim, d_state)

    n_chunks = math.ceil(T / chunk_size)
    for c in range(n_chunks):
        t0, t1 = c * chunk_size, min((c + 1) * chunk_size, T)
        L = t1 - t0

        delta_c = delta[:, t0:t1]  # (B, L, H)
        B_c = B_seq[:, t0:t1]  # (B, L, N)
        C_c = C_seq[:, t0:t1]  # (B, L, N)
        x_c = x_in_h[:, t0:t1]  # (B, L, H, P)

        logA = delta_c * A.view(1, 1, n_heads)  # (B, L, H)
        logA_cumsum = torch.cumsum(logA, dim=1)  # (B, L, H)

        carry = (
            torch.exp(logA_cumsum).unsqueeze(-1).unsqueeze(-1) * state.unsqueeze(1)
        )  # (B, L, H, P, N)

        # Intra-chunk decay: (B, L, L, H) -- no P/N dims, unlike Phase 1/2's
        # mamba_chunked_scan. Same "subtract cumsums before exponentiating"
        # safety as before.
        decay = logA_cumsum.unsqueeze(2) - logA_cumsum.unsqueeze(1)  # (B, L_i, L_k, H)
        causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        decay = decay.masked_fill(~causal_mask.view(1, L, L, 1), float("-inf"))
        decay = torch.exp(decay)

        deltaB_x = (
            delta_c.view(Bsz, L, n_heads, 1, 1)
            * B_c.view(Bsz, L, 1, 1, d_state)
            * x_c.unsqueeze(-1)
        )  # (B, L, H, P, N)
        intra = torch.einsum("blkh,bkhpn->blhpn", decay, deltaB_x)  # (B, L, H, P, N)

        state_seq = carry + intra  # state_i for each i in this chunk
        y[:, t0:t1] = torch.einsum("blhpn,bln->blhp", state_seq, C_c)

        state = state_seq[:, -1]  # end-of-chunk state, carried to the next chunk

    return y.reshape(Bsz, T, d_inner) + D * x_in


class MambaMixer(nn.Module):
    """Selective state-space mixer (Mamba's S6), extending S4DMixer's
    diagonal SSM with input-dependent ("selective") Delta/B/C.

    A stays a fixed, learned per-channel diagonal parameter, exactly like
    S4D. What changes is that Delta, B and C are now functions of the input
    at each timestep rather than fixed learned tensors -- the "selectivity"
    that lets the model decide, content-dependently, what to remember or
    forget. That same input-dependence breaks S4D's FFT-convolution
    shortcut (the recurrence is no longer linear time-invariant), which is
    exactly why Mamba needed a hardware-aware parallel scan kernel instead.
    Here it's a plain sequential scan over T -- correct, but O(T)
    sequential steps with no parallelism across time; a real fused/parallel
    scan is the natural Stage-4-style follow-up once this is measured.

    Also includes the surrounding block shape from the Mamba paper (input
    projection + causal depthwise conv + SiLU gating) since that structure
    is part of the trick, not incidental to it.
    """

    def __init__(self, cfg):
        super().__init__()
        n_embd = cfg["n_embd"]
        d_state = cfg["d_state"]
        d_inner = cfg["expand"] * n_embd
        d_conv = cfg["d_conv"]
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(n_embd, 2 * d_inner, bias=cfg["bias"])
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, bias=True)
        self.x_proj = nn.Linear(d_inner, d_inner + 2 * d_state, bias=False)

        A_log = torch.log(torch.arange(1, d_state + 1).float().expand(d_inner, d_state).clone())
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, n_embd, bias=cfg["bias"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        B, T, _ = x.shape
        x_in, z = self.in_proj(x).chunk(2, dim=-1)  # each (B, T, d_inner)

        x_in = x_in.transpose(1, 2)  # (B, d_inner, T)
        x_in = F.pad(x_in, (self.d_conv - 1, 0))
        x_in = F.silu(self.conv1d(x_in))
        x_in = x_in.transpose(1, 2)  # (B, T, d_inner)

        delta, B_seq, C_seq = torch.split(
            self.x_proj(x_in), [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(delta)  # (B, T, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        state = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            delta_t = delta[:, t]  # (B, d_inner)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            deltaB_x = (
                delta_t.unsqueeze(-1) * B_seq[:, t].unsqueeze(1) * x_in[:, t].unsqueeze(-1)
            )  # (B, d_inner, d_state)
            state = A_bar * state + deltaB_x
            y_t = torch.einsum("bdn,bn->bd", state, C_seq[:, t]) + self.D * x_in[:, t]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)

        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))


def build_attention(cfg):
    if cfg["attn_type"] == "full":
        return CausalSelfAttention(cfg)
    if cfg["attn_type"] == "local":
        return LocalCausalSelfAttention(cfg)
    if cfg["attn_type"] == "dilated":
        return DilatedCausalSelfAttention(cfg)
    if cfg["attn_type"] == "s4":
        return S4DMixer(cfg)
    if cfg["attn_type"] == "mamba":
        return MambaMixer(cfg)
    raise NotImplementedError(
        f"attn_type={cfg['attn_type']!r} not implemented yet — Stage 2 work"
    )


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"], bias=cfg["bias"])
        self.proj = nn.Linear(4 * cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = build_attention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def resolve_layer_cfg(cfg, i):
    """Effective per-layer cfg. cfg["layer_recipe"][i], if present, overrides
    attn_type (and any per-layer hyperparams) for layer i -- this is the seam
    for hybrid architectures (Jamba/Griffin/Zamba-style interleaving of
    different mixer types across depth). Without layer_recipe every layer
    resolves to cfg unchanged: today's single-mixer-type behavior, exactly
    as before.
    """
    if "layer_recipe" not in cfg:
        return cfg
    spec = cfg["layer_recipe"][i]
    if isinstance(spec, str):
        spec = {"attn_type": spec}
    return {**cfg, **spec}


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg["block_size"]

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList(
            [Block(resolve_layer_cfg(cfg, i)) for i in range(cfg["n_layer"])]
        )
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)

        self.tok_emb.weight = self.head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
