"""
Field 3D — a three.js cutaway of one paddy plot, as a Streamlit component.

The frontend is plain ES modules with three.js vendored under frontend/vendor/,
so there is no npm build step: Streamlit serves the folder as-is, locally and
on Streamlit Cloud. Restore third-party files with scripts/fetch_3d_assets.py.
"""
from __future__ import annotations

from pathlib import Path

from .state import SCENE_KEYS, build_scene_state

_FRONTEND = Path(__file__).resolve().parent / "frontend"
_MODELS = _FRONTEND / "assets" / "models"
_component = None


def field3d(state: dict, key: str = "field3d", height: int = 620):
    """Render the diorama. A stable `key` keeps the scene alive across reruns,
    so new state animates in instead of rebuilding the WebGL context."""
    global _component
    if _component is None:
        import streamlit.components.v1 as components
        _component = components.declare_component("field3d", path=str(_FRONTEND))
    # Optional GLBs (farmer/palm/pump) are listed so the frontend never has to
    # probe for missing files; see assets/CREDITS.md.
    models = sorted(p.name for p in _MODELS.glob("*.glb")) if _MODELS.is_dir() else []
    return _component(state=state, models=models, height=height, key=key, default=None)


__all__ = ["field3d", "build_scene_state", "SCENE_KEYS"]
