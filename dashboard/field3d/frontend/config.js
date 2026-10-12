// Shared constants for the Field 3D diorama.
//
// Units: 1 scene unit = 1 m horizontally. Water and soil depths are tiny next
// to a 12 m plot, so every *depth* in mm goes through mmY(), which exaggerates
// it by VEX. Plants are drawn at roughly true height. The HUD states VEX.

export const VEX = 4;
export const mmY = (mm) => (mm * VEX) / 1000;

export const SOIL_DEPTH_MM = 400;            // shown below the surface
export const BOTTOM = mmY(-SOIL_DEPTH_MM);   // underside of the soil block
export const PUDDLED_MM = 150;               // puddled topsoil
export const PAN_MM = 200;                   // plough pan base
export const TUBE_DEPTH_MM = 300;            // AWD tube reaches the wl floor
export const WL_MIN_MM = -300;               // config hydrology.wl_floor_mm

// Plot footprint. Bunds run along the back (−z) and left (−x) edges; the front
// (+z) and right (+x) faces are the cut planes through the field.
export const PLOT = { x0: -6, x1: 6, z0: -4, z1: 4 };
export const BUND_W = 0.6;
export const FIELD = { x0: PLOT.x0 + BUND_W, x1: PLOT.x1, z0: PLOT.z0 + BUND_W, z1: PLOT.z1 };

// Positions of the instruments on the front cut face.
export const TUBE = { x: 0.9, r: 0.09 };
export const POST = { x: 0.25, z: 3.55 };
export const FAN = { x0: 3.3, dx: 0.66, z: PLOT.z1 + 0.06 };
export const PUMP = { x: -3.4, z: -6.5 };
export const WEIR = { z: 0.6, w: 0.7 };

// Palette — mirrors dashboard/theme.py so the 3D view matches the deck.
export const C = {
  ink: '#16211F', water: '#1F6F6B', rice: '#3F6B34', rust: '#A33B2A',
  amber: '#8A6614', sky: '#2B6CA3', fan: '#5ED3C7',
};
export const ACTION_COLOUR = {
  MAINTAIN: C.rice, IRRIGATE: C.sky, DELAY: C.amber, DRAIN: C.water, INSPECT: C.rust,
};
export const ACTION_GLOSS = {
  MAINTAIN: 'No action needed',
  IRRIGATE: 'Add water now',
  DELAY: 'Hold — rain is coming',
  DRAIN: 'Start an AWD dry cycle',
  INSPECT: 'Automation blocked',
};

// Piecewise-linear lookup over [[x, value], ...]; value may be a number or [r,g,b].
export function keyframes(table, x) {
  if (x <= table[0][0]) return table[0][1];
  for (let i = 1; i < table.length; i++) {
    const [x1, v1] = table[i];
    if (x <= x1) {
      const [x0, v0] = table[i - 1];
      const t = (x - x0) / (x1 - x0);
      return Array.isArray(v0) ? v0.map((a, k) => a + (v1[k] - a) * t) : v0 + (v1 - v0) * t;
    }
  }
  return table[table.length - 1][1];
}

export const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
