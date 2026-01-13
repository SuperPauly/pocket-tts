import math

import torch
from torch import nn


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: int | torch.Tensor = 0,
    max_period: int | float = 10_000,
    freqs: torch.Tensor | None = None,
):
    """
    Args:
        q (torch.Tensor): Queries, shape `[B, T, H, D]`.
        k (torch.Tensor): Keys, shape `[B, T, H, D]`.
        offset (int): Current offset, e.g. when streaming.
        max_period (float): Maximum period for the cos and sin.
    """

    B, T, H, D = q.shape
    Bk, Tk, Hk, Dk = k.shape
    assert (B, T, D) == (Bk, Tk, Dk)
    assert D > 0
    assert D % 2 == 0
    assert max_period > 0

    if freqs is None:
        ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))

    # could be optimized in one call
    ts = torch.arange(T, device=q.device, dtype=torch.float32)
    ts += offset
    ts = ts.view(-1, 1, 1)

    q = q.view(B, T, H, D // 2, 2)
    k = k.view(B, T, Hk, D // 2, 2)

    # convention is `r` suffix is real part, `i` is imaginary.
    qr = q[..., 0].float()
    qi = q[..., 1].float()

    kr = k[..., 0].float()
    ki = k[..., 1].float()

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr

    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

    return qo.view(B, T, H, D), ko.view(B, T, Hk, D)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    Args:
        max_period (float): Maximum period of the rotation frequencies.
    """

    def __init__(self, max_period: float | int = 10000.0):
        super().__init__()
        self.max_period = max_period
        self.register_buffer("_cached_freqs", torch.empty(0), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor | int):
        """Apply rope rotation to query or key tensor."""
        d = q.shape[-1]
        if (
            self._cached_freqs.numel() != d // 2
            or self._cached_freqs.device != q.device
        ):
            ds = torch.arange(d // 2, device=q.device, dtype=torch.float32)
            freqs = torch.exp(ds * (-math.log(self.max_period) * 2 / d))
            self._cached_freqs = freqs
        return apply_rope(q, k, offset, self.max_period, freqs=self._cached_freqs)
