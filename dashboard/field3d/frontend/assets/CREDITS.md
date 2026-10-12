# Field 3D — third-party assets

Everything here is fetched by `scripts/fetch_3d_assets.py` (pinned URLs) and
committed so the dashboard needs no network at runtime.

| File | Source | Licence |
|---|---|---|
| `../vendor/three.module.js`, `../vendor/addons/**` | [three.js r170](https://github.com/mrdoob/three.js/tree/r170) via jsDelivr | MIT (`../vendor/LICENSE`) |
| `textures/mud_wet_diff.jpg`, `textures/mud_wet_nor.jpg` | [Brown Mud 02](https://polyhaven.com/a/brown_mud_02), Poly Haven, 1k | CC0 |
| `textures/mud_dry_diff.jpg`, `textures/mud_dry_nor.jpg` | [Mud Cracked Dry 03](https://polyhaven.com/a/mud_cracked_dry_03), Poly Haven, 1k | CC0 |
| `textures/grass_diff.jpg` | [Leafy Grass](https://polyhaven.com/a/leafy_grass), Poly Haven, 1k | CC0 |
| `hdri/sky_1k.hdr` | [Kloofendal 48d Partly Cloudy (Pure Sky)](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky), Poly Haven, 1k | CC0 |

Everything else in the scene — rice plants, water, soil strata, bunds, the AWD
tube, ESP32 post, pump, weir, palms, clouds and rain — is procedural geometry
in the `.js` files next to this folder.

## Optional models (`models/`)

Not shipped. If you add any of these, list them here with author, URL and licence
(CC-BY models need attribution in this table).

| File | Suggested source | Licence |
|---|---|---|
| `models/farmer.glb` | a Quaternius animated character | CC0 |
| `models/palm.glb` | a coconut palm from poly.pizza / Quaternius | CC0 / CC-BY |
| `models/pump.glb` | a water / diesel pump from poly.pizza or Sketchfab | check per model |
