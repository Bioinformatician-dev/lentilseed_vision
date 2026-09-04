"""
Outline shape: elliptic Fourier descriptors, the shape atlas, and PCA.

EFD turns a closed outline into a fixed-length numeric vector, so seeds of
different sizes and orientations become comparable. Normalisation removes size,
rotation and starting point, leaving shape alone.

PCA is done with a plain SVD rather than scikit-learn, to keep the deployment
dependency list at what the app already needs.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------

def elliptic_fourier(contour: np.ndarray, harmonics: int = 10,
                     normalize: bool = True) -> np.ndarray | None:
    c = np.asarray(contour).reshape(-1, 2).astype(float)
    if len(c) < 8:
        return None

    dxy = np.diff(np.vstack([c, c[:1]]), axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    dt[dt == 0] = 1e-9
    t = np.concatenate([[0.0], np.cumsum(dt)])
    T = t[-1]
    phi = 2 * np.pi * t / T

    coeffs = np.zeros((harmonics, 4))
    for n in range(1, harmonics + 1):
        cn = T / (2 * n ** 2 * np.pi ** 2)
        cosd = np.cos(n * phi[1:]) - np.cos(n * phi[:-1])
        sind = np.sin(n * phi[1:]) - np.sin(n * phi[:-1])
        coeffs[n - 1] = [
            cn * np.sum(dxy[:, 0] / dt * cosd), cn * np.sum(dxy[:, 0] / dt * sind),
            cn * np.sum(dxy[:, 1] / dt * cosd), cn * np.sum(dxy[:, 1] / dt * sind),
        ]

    if not normalize:
        return coeffs

    # Rotate to the major axis of the first harmonic, align the starting point,
    # then divide out size.
    a1, b1, c1, d1 = coeffs[0]
    theta = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1 ** 2 - b1 ** 2 + c1 ** 2 - d1 ** 2)
    for n in range(1, harmonics + 1):
        M = coeffs[n - 1].reshape(2, 2)
        R = np.array([[np.cos(n * theta), -np.sin(n * theta)],
                      [np.sin(n * theta), np.cos(n * theta)]])
        coeffs[n - 1] = M.dot(R).flatten()

    psi = np.arctan2(coeffs[0, 2], coeffs[0, 0])
    P = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])
    for n in range(1, harmonics + 1):
        coeffs[n - 1] = P.dot(coeffs[n - 1].reshape(2, 2)).flatten()

    if abs(coeffs[0, 0]) > 1e-12:
        coeffs = coeffs / abs(coeffs[0, 0])
    return coeffs


def efd_reconstruct(coeffs: np.ndarray, n_points: int = 180):
    """Draw an outline back from its descriptors."""
    t = np.linspace(0, 1.0, n_points)
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    for n in range(coeffs.shape[0]):
        k = 2 * (n + 1) * np.pi * t
        x += coeffs[n, 0] * np.cos(k) + coeffs[n, 1] * np.sin(k)
        y += coeffs[n, 2] * np.cos(k) + coeffs[n, 3] * np.sin(k)
    return x, y


# ---------------------------------------------------------------------------
# atlas & PCA
# ---------------------------------------------------------------------------

def descriptor_matrix(store: dict) -> tuple[np.ndarray, list]:
    """Stack the per-seed descriptors into rows, dropping the failures."""
    keys = [k for k, v in store.items() if v is not None]
    if not keys:
        return np.zeros((0, 0)), []
    X = np.stack([np.asarray(store[k]).ravel() for k in keys])
    # The first harmonic is fixed by normalisation and carries no information.
    return X[:, 4:], keys


def mean_shape(store: dict) -> np.ndarray | None:
    mats = [v for v in store.values() if v is not None]
    if not mats:
        return None
    return np.mean(np.stack(mats), axis=0)


def pca(X: np.ndarray, n_components: int = 4) -> dict:
    """Centred PCA by SVD. Returns scores, loadings and explained variance."""
    if X.size == 0 or X.shape[0] < 3:
        return {}
    n_components = int(min(n_components, X.shape[0] - 1, X.shape[1]))
    if n_components < 1:
        return {}

    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / max(1, (X.shape[0] - 1))
    total = var.sum()

    return {
        "scores": U[:, :n_components] * S[:n_components],
        "components": Vt[:n_components],
        "explained": var[:n_components] / total if total else var[:n_components],
        "explained_cum": np.cumsum(var[:n_components] / total) if total else None,
        "mean": mu,
        "n_components": n_components,
    }


def shape_along_pc(model: dict, harmonics: int, pc: int = 0, sd: float = 2.0,
                   n_points: int = 180):
    """
    The outline at +/- sd along one principal component.

    Turns an abstract axis into two pictures: this is what a seed at the low
    end of PC1 looks like, and this is the high end.
    """
    if not model:
        return None
    scores = model["scores"][:, pc]
    spread = float(scores.std()) * sd
    out = []
    for offset in (-spread, +spread):
        vec = model["mean"] + offset * model["components"][pc]
        coeffs = np.zeros((harmonics, 4))
        coeffs[0] = [1.0, 0.0, 0.0, 0.0]      # the harmonic normalisation removed
        coeffs[1:] = vec.reshape(-1, 4)
        out.append(efd_reconstruct(coeffs, n_points))
    return out
