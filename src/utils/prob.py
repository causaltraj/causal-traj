import numpy as np
import torch
import torch.nn.functional as F


def categorical_entropy(log_pi):
    k = log_pi.shape[-1]
    pi = torch.softmax(log_pi, dim=-1)
    return -(pi * (torch.log(pi + 1e-8))).sum(dim=-1).mean() / np.log(float(k))


def _cholesky_from_packed(L_packed, eps=1e-5):
    """
    L_packed: [..., 3] -> [..., 2, 2] lower-triangular with positive diag via softplus.
      order: [l11_raw, l21, l22_raw]
    """
    l11_raw, l21, l22_raw = L_packed.unbind(dim=-1)
    l11 = F.softplus(l11_raw) + eps
    l22 = F.softplus(l22_raw) + eps
    zeros = torch.zeros_like(l11)
    L = torch.stack(
        [
            torch.stack([l11, zeros], dim=-1),
            torch.stack([l21, l22], dim=-1),
        ],
        dim=-2,
    )  # [..., 2, 2]
    return L


def _sanitize_L_raw(L_raw, eps=1e-5):
    """
    Force lower-triangular and positive diag on provided 2x2 matrices.
    L_raw: [..., 2, 2]
    """
    # zero out upper triangle
    L = torch.zeros_like(L_raw)
    L[..., 0, 0] = L_raw[..., 0, 0]
    L[..., 1, 0] = L_raw[..., 1, 0]
    # positive diag via softplus
    L[..., 0, 0] = F.softplus(L[..., 0, 0]) + eps
    L[..., 1, 1] = F.softplus(L_raw[..., 1, 1]) + eps
    return L


def nll_mog_block2d(
    y,  # [B, 11, 2]
    mu,  # [B, K, 11, 2]
    log_pi,  # [B, K]
    L_packed=None,  # [B, K, 11, 3], optional preferred input
    L_raw=None,  # [B, K, 11, 2, 2], alternative
    eps=1e-5,
    reduction="mean",
):
    """
    Mixture of Gaussians NLL with block-diagonal covariance:
    per component k: Sigma_k = blockdiag(Sigma_{k,1}, ..., Sigma_{k,11}),
    each block Sigma_{k,p} = L_{k,p} L_{k,p}^T with 2x2 Cholesky L_{k,p}.
    """
    assert (L_packed is not None) ^ (L_raw is not None), "Provide exactly one of L_packed or L_raw"
    B, K, P, two = mu.shape
    assert two == 2, "mu must be [B,K,A,2]"
    assert y.shape == (B, P, two)
    assert log_pi.shape == (B, K)

    # Build Cholesky factors per block
    if L_packed is not None:
        assert L_packed.shape == (B, K, P, 3)
        L = _cholesky_from_packed(L_packed, eps=eps)  # [B,K,11,2,2]
    else:
        assert L_raw.shape == (B, K, P, 2, 2)
        L = _sanitize_L_raw(L_raw, eps=eps)  # [B,K,11,2,2]

    # Residuals per block
    r = (y.unsqueeze(1) - mu).unsqueeze(-1)  # [B,K,11,2,1]

    # Solve L z = r -> z = L^{-1} r  (batched triangular solve)
    z = torch.linalg.solve_triangular(L, r, upper=False)  # [B,K,11,2,1]
    quad_blocks = (z.squeeze(-1) ** 2).sum(dim=-1)  # [B,K,11], per 2D block

    # log |Sigma_block| = 2 * (log l11 + log l22) for each 2x2 block
    diag = torch.stack([L[..., 0, 0], L[..., 1, 1]], dim=-1)  # [B,K,11,2]
    logdet_blocks = 2.0 * torch.log(diag).sum(dim=-1)  # [B,K,11]

    # Sum across 11 players to get total per-component terms
    quad = quad_blocks.sum(dim=-1)  # [B,K]
    logdet = logdet_blocks.sum(dim=-1)  # [B,K]

    D_total = P * 2
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, device=y.device))
    log_pdf_k = -0.5 * (D_total * log2pi + logdet + quad)  # [B,K]

    # Mix over components (stable)
    log_pi_sm = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)  # [B,K]
    log_prob = torch.logsumexp(log_pi_sm + log_pdf_k, dim=-1)  # [B]
    nll = -log_prob

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    elif reduction == "none":
        return nll
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")


def sample_mog_block2d(
    mu,  # [B, K, 11, 2]   means
    log_pi,  # [B, K]          unnormalized mixture logits
    L_packed=None,  # [B, K, 11, 3]  (preferred)
    L_raw=None,  # [B, K, 11, 2, 2] (alternative)
    S: int = 1,
    eps=1e-5,
    mean_sampling=False,
    pi_temperature=1.0,
):
    """
    Draw S samples from a block-diagonal MoG with 11 independent 2x2 full-cov blocks.

    Returns:
      samples: [B, S, 11, 2]
      k_idx:   [B, S]  sampled component indices
    """
    assert (L_packed is not None) ^ (L_raw is not None), "Provide exactly one of L_packed or L_raw"
    B, K, P, two = mu.shape
    assert two == 2, "mu must be [B,K,A,2]"
    assert log_pi.shape == (B, K)

    device = mu.device
    dtype = mu.dtype

    # Build valid 2x2 Cholesky factors per block
    if L_packed is not None:
        assert L_packed.shape == (B, K, P, 3)
        L = _cholesky_from_packed(L_packed, eps=eps)  # [B,K,11,2,2]
    else:
        assert L_raw.shape == (B, K, P, 2, 2)
        L = _sanitize_L_raw(L_raw, eps=eps)  # [B,K,11,2,2]

    if K == 1:
        mu_sel = mu[:, :1, ...].expand(B, S, P, two)  # [B,S,11,2]
        L_sel = L[:, :1, ...].expand(B, S, P, two, two)  # [B,S,11,2,2]
        k_idx = torch.zeros(B, S, dtype=torch.long, device=mu.device)
    else:
        # Mixture component sampling
        pi = torch.softmax(log_pi / pi_temperature, dim=-1)  # [B,K]
        cat = torch.distributions.Categorical(pi)
        # sample S components independently per batch item
        k_idx = cat.sample((S,)).transpose(0, 1)  # [B,S]

        # Gather chosen component parameters
        # expand K-dim gather with an extra S axis
        mu_sel = (
            mu.unsqueeze(1)
            .gather(
                dim=2, index=k_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, S, 1, P, two)
            )
            .squeeze(2)
        )  # [B,S,11,2]

        if mean_sampling:
            return mu_sel, k_idx

        L_sel = (
            L.unsqueeze(1)
            .gather(
                dim=2,
                index=k_idx.unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(B, S, 1, P, two, two),
            )
            .squeeze(2)
        )  # [B,S,11,2,2]

    # Sample standard normals per block (2D), then transform by Cholesky
    eps = torch.randn(B, S, P, two, 1, device=device, dtype=dtype)  # [B,S,11,2,1]
    r = torch.matmul(L_sel, eps).squeeze(-1)  # [B,S,11,2]

    samples = mu_sel + r  # [B,S,11,2]
    return samples, k_idx
