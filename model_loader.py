"""
model_loader.py — put `best.pt` in front of the app with nothing for the user to do.

`best.pt` is committed beside app.py, so a fresh clone runs straight away and
the app never asks anyone to upload a model. This module resolves the weights,
in order:

  1. `best.pt` sitting next to app.py            — the committed default
  2. `LENTIL_WEIGHTS_PATH` env var                — Docker / your own server, weights
                                                    baked into the image or mounted
  3. `weights_url` in Streamlit secrets, or the
     `LENTIL_WEIGHTS_URL` env var                 — Streamlit Cloud / Spaces: the file
                                                    is downloaded once and cached on disk

For (3), host the file somewhere that serves raw bytes over HTTPS. A GitHub Release
asset is the usual choice — it lives outside git history, so it doesn't bloat the
clone, and there's no size ceiling to worry about at 19 MB:

    gh release create v1.0 best.pt --title "v1.0 weights"

then in `.streamlit/secrets.toml` (which is gitignored):

    weights_url = "https://github.com/<user>/<repo>/releases/download/v1.0/best.pt"

Hugging Face Hub works identically — use the `resolve/main/best.pt` URL.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).parent
BUNDLED = APP_DIR / "best.pt"
CACHE_DIR = Path(tempfile.gettempdir()) / "lentil_weights"

# 15 min is generous for ~20 MB, and stops a dead host from hanging the app forever.
DOWNLOAD_TIMEOUT = 900
CHUNK = 1 << 20  # 1 MiB


class WeightsError(RuntimeError):
    """Weights could not be resolved — the message is safe to show the user."""


def _secret(name: str) -> str:
    """Read a Streamlit secret, tolerating the common case of no secrets.toml at all."""
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:  # FileNotFoundError, StreamlitSecretNotFoundError, ...
        return ""


def weights_url() -> str:
    return _secret("weights_url") or os.environ.get("LENTIL_WEIGHTS_URL", "")


def _cached_path(url: str) -> Path:
    # Key on the URL so switching releases re-downloads instead of serving stale weights.
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.pt"


def _download(url: str, dest: Path) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    bar = st.progress(0.0, text="Downloading model weights\u2026")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "lentil-seedvision"})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            with open(part, "wb") as fh:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        bar.progress(
                            min(got / total, 1.0),
                            text=f"Downloading model weights\u2026 "
                                 f"{got / 1e6:.0f} / {total / 1e6:.0f} MB",
                        )
        if got == 0:
            raise WeightsError("The weights URL returned an empty file.")
        # Only move into place once the whole file landed, so an interrupted
        # download can never be mistaken for a valid cached copy.
        part.replace(dest)
    except urllib.error.HTTPError as e:
        raise WeightsError(
            f"Weights download failed ({e.code} {e.reason}). "
            "Check the URL is a direct download and the release/repo is public."
        ) from e
    except urllib.error.URLError as e:
        raise WeightsError(f"Could not reach the weights URL: {e.reason}") from e
    finally:
        bar.empty()
        part.unlink(missing_ok=True)


def resolve_weights(dev_path: str = "") -> str:
    """Return a local filesystem path to the weights, downloading them if needed."""
    if dev_path.strip():
        p = Path(dev_path.strip()).expanduser()
        if not p.exists():
            raise WeightsError(f"No weights file at {p}")
        return str(p)

    if BUNDLED.exists():
        return str(BUNDLED)

    env_path = os.environ.get("LENTIL_WEIGHTS_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if not p.exists():
            raise WeightsError(f"LENTIL_WEIGHTS_PATH points at a missing file: {p}")
        return str(p)

    url = weights_url()
    if not url:
        raise WeightsError(
            "No model weights configured. Put `best.pt` next to app.py for local use, "
            "or set `weights_url` in .streamlit/secrets.toml to a direct download link "
            "(see model_loader.py for the one-line `gh release create` recipe)."
        )

    dest = _cached_path(url)
    if not (dest.exists() and dest.stat().st_size > 0):
        _download(url, dest)
    return str(dest)


def describe_source() -> tuple[str, str]:
    """(label, detail) for the sidebar status line — no filesystem writes."""
    if BUNDLED.exists():
        return "Model loaded", "best.pt, bundled with the app"
    env_path = os.environ.get("LENTIL_WEIGHTS_PATH", "").strip()
    if env_path:
        return "Model loaded", f"server path: {env_path}"
    url = weights_url()
    if url:
        cached = _cached_path(url).exists()
        host = url.split("/")[2] if "//" in url else url
        return "Model loaded", (f"downloaded from {host}" if cached else f"from {host}")
    return "No model", "no best.pt beside app.py and no weights_url set"
