"""Shared edge chamfer scoring (cost volume + particle filter).

Query and rendered edges are both derived via Sobel on surface-normal maps;
the query side is precomputed as a distance transform of ``pred_e``.
Score: ``exp(-mean_chamfer / tau)`` with ``tau = max(3, H/50)`` pixels.
"""

import torch
import torch.nn.functional as F

_MIN_EDGE_PX = 5
_EDGE_MAG_THRESH = 0.5


def edge_tau_pixels(height: int) -> float:
    """Exponential kernel scale in pixels for a render of height ``height``."""
    return max(3.0, height / 50.0)


def _chamfer_vs_dt(
    rendered_edges: torch.Tensor,
    edge_dt: torch.Tensor,
    *,
    tau: float,
    max_edt: float = float('inf'),
) -> torch.Tensor:
    """Mean DT distance on rendered edge pixels → exponential score in [0, 1]."""
    n = rendered_edges.shape[0]
    device = rendered_edges.device
    dt = edge_dt.to(device)
    if dt.dim() == 2:
        dt = dt.unsqueeze(0)
    if dt.shape[0] == 1 and n > 1:
        dt = dt.expand(n, -1, -1)
    elif dt.shape[0] != n:
        dt = dt.expand(n, -1, -1)

    n_edge = rendered_edges.float().sum(dim=(1, 2)).clamp(min=1)
    mean_chamfer = (dt * rendered_edges.float()).sum(dim=(1, 2)) / n_edge
    raw_scores = torch.exp(-mean_chamfer / tau)
    
    # Penalize poses that render very few expected edges (e.g. flooded mask or empty sky)
    pred_edges_relaxed = dt < 2.0
    n_pred_edges = pred_edges_relaxed.float().sum(dim=(1, 2)).clamp(min=1)
    overlap = (rendered_edges & pred_edges_relaxed).float().sum(dim=(1, 2))
    e_rec = overlap / n_pred_edges
    
    # A good pose typically has > 10% edge recall. Poses with tiny overlap (e.g. 1%) are heavily penalized.
    edge_conf = torch.clamp(e_rec / 0.10, max=1.0)
    scores = raw_scores * edge_conf
    
    # Hard constraint: completely restrict the search space for candidates without meaningful edges
    # Any candidate covering less than 5% of predicted edges gets a score of exactly 0.
    scores = torch.where(e_rec >= 0.05, scores, torch.zeros_like(scores))
    
    # Hard constraint: restrict candidates with worse EDT than the GPS prior
    scores = torch.where(mean_chamfer <= max_edt, scores, torch.zeros_like(scores))
    
    return torch.where(n_edge > _MIN_EDGE_PX, scores, torch.zeros_like(scores))


def score_mask_edges_vs_dt(
    masks_bhw: torch.Tensor,
    edge_dt: torch.Tensor,
    *,
    tau: float | None = None,
    max_edt: float = float('inf'),
) -> torch.Tensor:
    """Score batch of poses: silhouette boundary vs query edge distance transform.

    Uses a single Sobel pass on the rendered binary mask — no extra normal raster.

    Args:
        masks_bhw: (N, H, W) bool/float rendered silhouettes.
        edge_dt: (H, W) or (1, H, W) query edge DT at the same resolution.
        tau: optional override; default ``edge_tau_pixels(H)``.

    Returns:
        (N,) scores in [0, 1], higher = better edge alignment.
    """
    h = masks_bhw.shape[1]
    device = masks_bhw.device
    if tau is None:
        tau = edge_tau_pixels(h)

    m = masks_bhw.float().unsqueeze(1)
    kx = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(m, kx, padding=1)
    gy = F.conv2d(m, ky, padding=1)
    rendered_edges = (gx.abs() + gy.abs()) > 0.05
    return _chamfer_vs_dt(rendered_edges.squeeze(1), edge_dt, tau=tau)


def score_normal_edges_vs_dt(
    normals_bchw: torch.Tensor,
    edge_dt: torch.Tensor,
    *,
    tau: float | None = None,
    max_edt: float = float('inf'),
) -> torch.Tensor:
    """Score batch of poses: normal Sobel edges vs query edge distance transform.

    Args:
        normals_bchw: (N, 3, H, W) rendered camera-frame normals.
        edge_dt: (H, W) or (1, H, W) query edge DT at the same resolution.
        tau: optional override; default ``edge_tau_pixels(H)``.

    Returns:
        (N,) scores in [0, 1], higher = better edge alignment.
    """
    h = normals_bchw.shape[2]
    device = normals_bchw.device

    if tau is None:
        tau = edge_tau_pixels(h)

    kx = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(normals_bchw.float(), kx.expand(3, 1, 3, 3), groups=3, padding=1)
    gy = F.conv2d(normals_bchw.float(), ky.expand(3, 1, 3, 3), groups=3, padding=1)
    edge_mag = (gx ** 2 + gy ** 2 + 1e-8).sqrt().sum(dim=1)
    rendered_edges = edge_mag > _EDGE_MAG_THRESH
    return _chamfer_vs_dt(rendered_edges, edge_dt, tau=tau, max_edt=max_edt)
