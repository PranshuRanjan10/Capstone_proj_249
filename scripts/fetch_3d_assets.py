"""
Download the third-party files behind the Field 3D tab.

Everything lands in dashboard/field3d/frontend/ and is committed, so the
dashboard never touches a CDN at runtime (the demo works offline and on
Streamlit Cloud). Re-run this only to restore or upgrade the files.

    python scripts/fetch_3d_assets.py           # skip files already present
    python scripts/fetch_3d_assets.py --force   # re-download everything

Sources and licences are listed in dashboard/field3d/frontend/assets/CREDITS.md.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "dashboard" / "field3d" / "frontend"

THREE = "https://cdn.jsdelivr.net/npm/three@0.170.0"
PH_TEX = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k"
PH_HDR = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k"

# (url, destination relative to FRONT). Pinned: three.js by exact version,
# Poly Haven by asset slug and resolution.
FILES = [
    # three.js r170 (MIT) — the addons keep their relative layout because
    # GLTFLoader imports ../utils/BufferGeometryUtils.js
    (f"{THREE}/build/three.module.js",                    "vendor/three.module.js"),
    (f"{THREE}/examples/jsm/controls/OrbitControls.js",   "vendor/addons/controls/OrbitControls.js"),
    (f"{THREE}/examples/jsm/loaders/RGBELoader.js",       "vendor/addons/loaders/RGBELoader.js"),
    (f"{THREE}/examples/jsm/loaders/GLTFLoader.js",       "vendor/addons/loaders/GLTFLoader.js"),
    (f"{THREE}/examples/jsm/renderers/CSS2DRenderer.js",  "vendor/addons/renderers/CSS2DRenderer.js"),
    (f"{THREE}/examples/jsm/utils/BufferGeometryUtils.js", "vendor/addons/utils/BufferGeometryUtils.js"),
    (f"{THREE}/LICENSE",                                  "vendor/LICENSE"),
    # Poly Haven (CC0) — wet puddled surface, AWD-dried surface, bund grass
    (f"{PH_TEX}/brown_mud_02/brown_mud_02_diff_1k.jpg",            "assets/textures/mud_wet_diff.jpg"),
    (f"{PH_TEX}/brown_mud_02/brown_mud_02_nor_gl_1k.jpg",          "assets/textures/mud_wet_nor.jpg"),
    (f"{PH_TEX}/mud_cracked_dry_03/mud_cracked_dry_03_diff_1k.jpg", "assets/textures/mud_dry_diff.jpg"),
    (f"{PH_TEX}/mud_cracked_dry_03/mud_cracked_dry_03_nor_gl_1k.jpg", "assets/textures/mud_dry_nor.jpg"),
    (f"{PH_TEX}/leafy_grass/leafy_grass_diff_1k.jpg",              "assets/textures/grass_diff.jpg"),
    # Poly Haven (CC0) — sky, used for both background and image-based lighting
    (f"{PH_HDR}/kloofendal_48d_partly_cloudy_puresky_1k.hdr",      "assets/hdri/sky_1k.hdr"),
]


def fetch(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "CARE-Paddy asset fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    dest.write_bytes(data)
    return len(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    total, failed = 0, []
    for url, rel in FILES:
        dest = FRONT / rel
        if dest.exists() and not args.force:
            print(f"  skip  {rel}")
            total += dest.stat().st_size
            continue
        try:
            n = fetch(url, dest)
            total += n
            print(f"  ok    {rel}  ({n / 1e6:.2f} MB)")
        except Exception as exc:                              # pragma: no cover
            failed.append(rel)
            print(f"  FAIL  {rel}: {exc}", file=sys.stderr)

    print(f"\n{len(FILES) - len(failed)}/{len(FILES)} files, {total / 1e6:.1f} MB in {FRONT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
