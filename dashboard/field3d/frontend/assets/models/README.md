# Optional models

Drop any of these `.glb` files here; the scene picks them up on the next rerun
and falls back to procedural geometry for any that are missing.

- `farmer.glb` — a person, scaled to 1.65 m, standing on the back bund (an idle clip plays if present)
- `palm.glb` — a coconut palm, scaled to ~8 m, replacing every procedural palm
- `pump.glb` — a pump set, scaled to ~0.9 m, replacing the procedural pump body

Keep each under ~2 MB (Draco / glTF-Transform compress well) and record the
author, URL and licence in `../CREDITS.md`.
