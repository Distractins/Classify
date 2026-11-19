"""Deterministic synthetic stain shifts; these are not external-center validation."""
import hashlib
import numpy as np
from PIL import Image


def image_rng(name, seed):
    digest = hashlib.sha256(f"{seed}/{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def perturb_stain(image, method, strength, rng):
    """Strength controls concentration scaling; Macenko basis noise is strength/5."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    if method == "original" or strength == 0:
        return image.copy()
    if method == "hed":
        # Standard H/E/D optical-density vectors, using signed coefficients to
        # preserve the reconstruction residual rather than clipping it away.
        basis = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
        od = -np.log(np.clip(rgb, 1 / 255, 1))
        stains = od @ np.linalg.inv(basis)
        # H and E only; preserve the third residual/DAB channel.
        stains[..., :2] *= rng.uniform(1 - strength, 1 + strength, size=2)
        shifted = np.exp(-(stains @ basis))
    elif method == "macenko":
        od = -np.log(np.clip(rgb.reshape(-1, 3), 1 / 255, 1))
        tissue = od[np.all(od > 0.15, axis=1)]
        if len(tissue) < 10:
            raise ValueError("Insufficient tissue pixels for Macenko stain estimation")
        _, singular, vectors = np.linalg.svd(tissue, full_matrices=False)
        if singular[1] < 1e-6:
            raise ValueError("Degenerate Macenko stain plane")
        plane = vectors[:2].T.copy()
        if plane[:, 0].sum() < 0:
            plane[:, 0] *= -1
        projection = tissue @ plane
        angles = np.arctan2(projection[:, 1], projection[:, 0])
        low, high = np.percentile(angles, [1, 99])
        basis = plane @ np.array([[np.cos(low), np.cos(high)],
                                  [np.sin(low), np.sin(high)]])
        if basis[0, 0] < basis[0, 1]:
            basis = basis[:, ::-1]
        if np.linalg.cond(basis) > 1e4:
            raise ValueError("Ill-conditioned Macenko stain basis")
        concentrations = np.maximum(np.linalg.lstsq(basis, od.T, rcond=None)[0], 0)
        residual = od.T - basis @ concentrations
        shifted_basis = np.maximum(basis + rng.normal(0, strength / 5, basis.shape), 1e-6)
        shifted_basis /= np.linalg.norm(shifted_basis, axis=0)
        concentrations *= rng.uniform(1 - strength, 1 + strength, (2, 1))
        shifted = np.exp(-(shifted_basis @ concentrations + residual).T).reshape(rgb.shape)
    else:
        raise ValueError(f"Unknown stain method: {method}")
    if not np.isfinite(shifted).all():
        raise ValueError("Non-finite stain perturbation")
    return Image.fromarray(np.uint8(np.round(np.clip(shifted, 0, 1) * 255)))
