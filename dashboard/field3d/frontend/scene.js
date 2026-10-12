// The static world: soil block with cut-face strata, bunds, ponded water,
// surrounding paddies, canal, palms and lighting. Everything that moves with
// the field state is driven from update(); actions and instruments live in
// gauge.js and fx.js.
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import {
  BOTTOM, BUND_W, FIELD, PLOT, PUDDLED_MM, PAN_MM, TUBE, TUBE_DEPTH_MM, WEIR,
  mmY, keyframes, clamp,
} from './config.js';

// ── helpers ──────────────────────────────────────────────────────────────────
export function label(text, cls = '', center = [0.5, 0.5]) {
  const el = document.createElement('div');
  el.className = `lbl3d ${cls}`;
  el.textContent = text;
  const o = new CSS2DObject(el);
  o.center.set(center[0], center[1]);
  o.el = el;
  return o;
}

export function setLabel(o, text, visible = true) {
  if (o.el.textContent !== text) o.el.textContent = text;
  o.visible = visible;
  o.el.style.display = visible ? '' : 'none';
}

function tiled(tex, sx, sz, metresPerTile = 3) {
  const t = tex.clone();
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(sx / metresPerTile, sz / metresPerTile);
  t.needsUpdate = true;
  return t;
}

// A box spanning [x0,x1]×[yBot,yTop]×[z0,z1]; BoxGeometry face order is
// +x, −x, +y, −y, +z, −z.
function slab(x0, x1, yBot, yTop, z0, z1, top, side) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0, yTop - yBot, z1 - z0),
    [side, side, top, side, side, side]);
  m.position.set((x0 + x1) / 2, (yBot + yTop) / 2, (z0 + z1) / 2);
  m.receiveShadow = true;
  return m;
}

// ── cut-face strata shader ───────────────────────────────────────────────────
// Layers: puddled topsoil → plough pan → subsoil. Below the water table the
// soil is saturated (darker, bluer) with a capillary fringe above it. On the
// plot's front face it also draws the AWD tube section and hatches the zone
// below the stage's AWD floor.
const STRATA_VERT = /* glsl */`
  varying vec3 vW; varying vec3 vN;
  void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vW = w.xyz; vN = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * w;
  }`;

const STRATA_FRAG = /* glsl */`
  uniform sampler2D uTex;
  uniform float uWT, uFloor, uSensor, uTime, uInstr, uPuddled, uPan;
  uniform float uTubeX, uTubeR, uTubeBot;
  varying vec3 vW; varying vec3 vN;
  vec3 lin(vec3 c) { return pow(c, vec3(2.2)); }
  void main() {
    float y = vW.y;
    vec2 uv = (abs(vN.z) > 0.5 ? vW.xy : (abs(vN.x) > 0.5 ? vW.zy : vW.xz)) * vec2(0.35, 0.7);
    float grain = dot(texture2D(uTex, uv).rgb, vec3(0.333));

    vec3 top = lin(vec3(0.34, 0.26, 0.19));
    vec3 pan = lin(vec3(0.45, 0.39, 0.33));
    vec3 sub = lin(vec3(0.60, 0.45, 0.29));
    vec3 c = mix(sub, pan, smoothstep(uPan - 0.025, uPan + 0.025, y));
    c = mix(c, top, smoothstep(uPuddled - 0.04, uPuddled + 0.04, y));
    c *= 0.72 + 0.7 * smoothstep(0.0, 0.35, grain);

    // saturated zone, capillary fringe, and the water-table line itself
    float sat = 1.0 - smoothstep(uWT - 0.008, uWT + 0.008, y);
    c = mix(c, c * vec3(0.46, 0.54, 0.68), sat * 0.9);
    float fringe = (1.0 - sat) * (1.0 - smoothstep(uWT, uWT + 0.16, y));
    c = mix(c, c * vec3(0.72, 0.77, 0.84), fringe * 0.6);
    float wline = (1.0 - smoothstep(0.004, 0.014, abs(y - uWT))) * step(uWT, -0.01);
    c = mix(c, lin(vec3(0.45, 0.82, 0.98)), wline * (0.75 + 0.25 * sin(uTime * 3.0 + vW.x * 4.0)));

    if (uInstr > 0.5 && vN.z > 0.5) {
      // risk hatch below the stage's AWD floor
      float below = 1.0 - smoothstep(uFloor - 0.004, uFloor + 0.004, y);
      float hatch = step(0.8, fract((vW.x + y) * 3.0));
      c = mix(c, lin(vec3(0.70, 0.22, 0.14)), below * hatch * 0.1);

      // AWD field-water tube, sliced in half by the cut
      float dx = abs(vW.x - uTubeX);
      if (dx < uTubeR + 0.014 && y > uTubeBot - 0.014) {
        vec3 inner = y < uSensor ? lin(vec3(0.22, 0.55, 0.74)) : lin(vec3(0.07, 0.09, 0.09));
        float wall = step(uTubeR, dx) * step(fract(y * 14.0), 0.62) + step(y, uTubeBot);
        vec3 tc = mix(inner, lin(vec3(0.88, 0.90, 0.89)), clamp(wall, 0.0, 1.0));
        float mk = (1.0 - smoothstep(0.006, 0.016, abs(y - uSensor))) * step(dx, uTubeR);
        c = mix(tc, lin(vec3(0.98, 0.56, 0.12)), mk);
      }
    }
    float shade = 0.7 + 0.3 * max(dot(normalize(vN), normalize(vec3(0.35, 0.8, 0.6))), 0.0);
    gl_FragColor = vec4(c * shade * 1.5, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;

function strataMaterial(tex, shared, instruments) {
  return new THREE.ShaderMaterial({
    vertexShader: STRATA_VERT,
    fragmentShader: STRATA_FRAG,
    uniforms: {
      uTex: { value: tex },
      uWT: shared.uWT, uTime: shared.uTime, uFloor: shared.uFloor, uSensor: shared.uSensor,
      uInstr: { value: instruments ? 1 : 0 },
      uPuddled: { value: mmY(-PUDDLED_MM) }, uPan: { value: mmY(-PAN_MM) },
      uTubeX: { value: TUBE.x }, uTubeR: { value: TUBE.r },
      uTubeBot: { value: mmY(-TUBE_DEPTH_MM) },
    },
  });
}

// ── wet ↔ dry surface ────────────────────────────────────────────────────────
// One MeshStandardMaterial whose albedo and normal crossfade from puddled mud
// to AWD-cracked mud as the water table drops.
function surfaceMaterial(tex, uDry) {
  const m = new THREE.MeshStandardMaterial({
    map: tiled(tex.wetDiff, 12, 8, 3), normalMap: tiled(tex.wetNor, 12, 8, 3),
    roughness: 0.6, normalScale: new THREE.Vector2(0.9, 0.9),
  });
  const dryD = tiled(tex.dryDiff, 12, 8, 3);
  const dryN = tiled(tex.dryNor, 12, 8, 3);
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uDry = uDry;
    sh.uniforms.uDryMap = { value: dryD };
    sh.uniforms.uDryNor = { value: dryN };
    sh.fragmentShader = 'uniform float uDry; uniform sampler2D uDryMap; uniform sampler2D uDryNor;\n' +
      sh.fragmentShader
        .replace('#include <map_fragment>', `
          vec4 wetC = texture2D(map, vMapUv) * vec4(vec3(0.62), 1.0);
          vec4 dryC = texture2D(uDryMap, vMapUv);
          diffuseColor *= mix(wetC, dryC, uDry);`)
        .replace('texture2D( normalMap, vNormalMapUv )',
          'mix(texture2D( normalMap, vNormalMapUv ), texture2D( uDryNor, vNormalMapUv ), uDry)');
  };
  return m;
}

// Tileable ripple normal map, generated so no extra download is needed.
function rippleNormals(size = 128) {
  const h = (x, y) => {
    const u = (x / size) * Math.PI * 2, v = (y / size) * Math.PI * 2;
    return Math.sin(u * 3 + Math.sin(v * 2)) * 0.5 + Math.sin(v * 5 + u) * 0.3 +
      Math.sin((u + v) * 7) * 0.15;
  };
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = h(x + 1, y) - h(x - 1, y), dy = h(x, y + 1) - h(x, y - 1);
      const n = new THREE.Vector3(-dx * 2, -dy * 2, 1).normalize();
      const i = (y * size + x) * 4;
      data[i] = (n.x * 0.5 + 0.5) * 255; data[i + 1] = (n.y * 0.5 + 0.5) * 255;
      data[i + 2] = (n.z * 0.5 + 0.5) * 255; data[i + 3] = 255;
    }
  }
  const t = new THREE.DataTexture(data, size, size);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(5, 3.5);
  t.needsUpdate = true;
  return t;
}

// ── bunds ────────────────────────────────────────────────────────────────────
function bundGeometry(len, h) {
  const s = new THREE.Shape();
  s.moveTo(0, 0); s.lineTo(BUND_W, 0); s.lineTo(BUND_W - 0.16, h); s.lineTo(0.16, h); s.closePath();
  return new THREE.ExtrudeGeometry(s, { depth: len, bevelEnabled: false });
}

// ── coconut palm (the Cauvery delta skyline) ─────────────────────────────────
function palm(rng) {
  const g = new THREE.Group();
  const H = 6 + rng() * 3, lean = (rng() - 0.5) * 1.6;
  const curve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(0, 0, 0), new THREE.Vector3(lean * 0.2, H * 0.5, 0),
    new THREE.Vector3(lean, H, lean * 0.3));
  const trunk = new THREE.Mesh(new THREE.TubeGeometry(curve, 12, 0.13, 6),
    new THREE.MeshStandardMaterial({ color: '#7d6a52', roughness: 0.95 }));
  trunk.castShadow = true;
  g.add(trunk);

  const leaf = new THREE.PlaneGeometry(0.55, 3.2, 1, 8);
  leaf.translate(0, 1.6, 0);
  const p = leaf.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const y = p.getY(i);
    p.setX(i, p.getX(i) * (1 - (y / 3.2) * 0.8));
    p.setZ(i, -0.18 * y * y);                         // arch downward
  }
  leaf.computeVertexNormals();
  const fronds = [];
  for (let i = 0; i < 11; i++) {
    const f = leaf.clone();
    f.rotateX(-0.5 - rng() * 0.5);
    f.rotateY((i / 11) * Math.PI * 2 + rng() * 0.3);
    fronds.push(f);
  }
  const crown = new THREE.Mesh(mergeGeometries(fronds),
    new THREE.MeshStandardMaterial({ color: '#3f6b2a', side: THREE.DoubleSide, roughness: 0.8 }));
  crown.position.copy(curve.getPoint(1));
  crown.castShadow = true;
  g.add(crown);
  return g;
}

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── build ────────────────────────────────────────────────────────────────────
export function createWorld(scene, tex) {
  const shared = {
    uWT: { value: 0 }, uTime: { value: 0 }, uFloor: { value: mmY(-150) },
    uSensor: { value: 0 }, uDry: { value: 0 },
  };
  const plotStrata = strataMaterial(tex.wetDiff, shared, true);
  const landStrata = strataMaterial(tex.wetDiff, shared, false);
  const dark = new THREE.MeshStandardMaterial({ color: '#2b2219' });

  // lighting
  scene.add(new THREE.HemisphereLight('#d8ecff', '#6b5a3c', 0.45));
  const sun = new THREE.DirectionalLight('#fff4e0', 2.4);
  sun.position.set(10, 16, 9);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -11, right: 11, top: 11, bottom: -11, near: 1, far: 50 });
  sun.shadow.bias = -0.0005;
  scene.add(sun);

  // the plot's soil block
  const surface = surfaceMaterial(tex, shared.uDry);
  scene.add(slab(PLOT.x0, PLOT.x1, BOTTOM, 0, PLOT.z0, PLOT.z1, surface, plotStrata));

  // surrounding land: an L of paddies behind and to the left, cut at the same
  // planes so the whole landscape reads as one section.
  const nbMat = (sx, sz) => new THREE.MeshStandardMaterial({
    map: tiled(tex.grass, sx, sz, 4), color: '#9fbf6a', roughness: 0.95,
  });
  const neighbours = [];
  const add = (m) => { scene.add(m); return m; };
  const backTop = nbMat(120, 54);
  neighbours.push(backTop);
  add(slab(-60, 60, BOTTOM, 0, -60, -5.6, backTop, landStrata));
  const bankTop = new THREE.MeshStandardMaterial({ map: tiled(tex.grass, 120, 0.6, 3), color: '#a9b884' });
  add(slab(-60, 60, BOTTOM, 0, -4.6, PLOT.z0, bankTop, landStrata));
  add(slab(-60, 60, BOTTOM, -0.42, -5.6, -4.6, dark, landStrata));            // canal bed
  const leftTop = nbMat(52.4, 8);
  neighbours.push(leftTop);
  add(slab(-60, -7.6, BOTTOM, 0, PLOT.z0, PLOT.z1, leftTop, landStrata));
  add(slab(-6.6, PLOT.x0, BOTTOM, 0, PLOT.z0, PLOT.z1, bankTop, landStrata));
  add(slab(-7.6, -6.6, BOTTOM, -0.36, PLOT.z0, PLOT.z1, dark, landStrata));   // drain channel bed

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(260, 260),
    new THREE.MeshStandardMaterial({ map: tiled(tex.grass, 260, 260, 5), color: '#7d9a5a', roughness: 1 }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = BOTTOM - 0.002;
  ground.receiveShadow = true;
  scene.add(ground);

  // neighbouring bunds — a low grid that makes the land read as paddies
  const nb = [];
  for (let x = -54; x <= 54; x += 12) nb.push(new THREE.BoxGeometry(0.35, 0.07, 54).translate(x, 0.035, -32.8));
  for (let z = -56; z <= -10; z += 10) nb.push(new THREE.BoxGeometry(120, 0.07, 0.35).translate(0, 0.035, z));
  for (let x = -55; x <= -10; x += 11) nb.push(new THREE.BoxGeometry(0.35, 0.07, 8).translate(x, 0.035, 0));
  scene.add(new THREE.Mesh(mergeGeometries(nb), new THREE.MeshStandardMaterial({ color: '#6f7a4a', roughness: 1 })));

  // canal + drain channel water
  const chanMat = new THREE.MeshStandardMaterial({ color: '#4c7f86', roughness: 0.15, metalness: 0.1 });
  const canal = new THREE.Mesh(new THREE.PlaneGeometry(120, 1.0), chanMat);
  canal.rotation.x = -Math.PI / 2; canal.position.set(0, -0.14, -5.1);
  scene.add(canal);
  const drain = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 8), chanMat);
  drain.rotation.x = -Math.PI / 2; drain.position.set(-7.1, -0.2, 0);
  scene.add(drain);

  // the plot's bunds (back, and left split by the drainage weir)
  const bundMat = new THREE.MeshStandardMaterial({ map: tiled(tex.grass, 2, 2, 1.5), color: '#9a9466', roughness: 1 });
  const bunds = new THREE.Group();
  scene.add(bunds);
  let bundBuiltFor = null;
  function buildBunds(bundMm) {
    bunds.clear();
    const h = mmY(bundMm) + 0.1;
    const back = new THREE.Mesh(bundGeometry(PLOT.x1 - PLOT.x0, h), bundMat);
    back.geometry.rotateY(Math.PI / 2);
    back.position.set(PLOT.x0, 0, PLOT.z0 + BUND_W);
    const l1 = new THREE.Mesh(bundGeometry(WEIR.z - WEIR.w / 2 - PLOT.z0, h), bundMat);
    l1.position.set(PLOT.x0, 0, PLOT.z0);
    const l2 = new THREE.Mesh(bundGeometry(PLOT.z1 - (WEIR.z + WEIR.w / 2), h), bundMat);
    l2.position.set(PLOT.x0, 0, WEIR.z + WEIR.w / 2);
    for (const m of [back, l1, l2]) { m.castShadow = m.receiveShadow = true; bunds.add(m); }
    bundBuiltFor = bundMm;
  }
  buildBunds(100);

  // ponded water — a box so the cut faces show the water column
  const waterTex = rippleNormals();
  const water = new THREE.Mesh(
    new THREE.BoxGeometry(FIELD.x1 - FIELD.x0 - 0.2, 1, FIELD.z1 - FIELD.z0 - 0.1),
    new THREE.MeshPhysicalMaterial({
      color: '#5d9aa0', transparent: true, opacity: 0.6, roughness: 0.06,
      normalMap: waterTex, normalScale: new THREE.Vector2(0.18, 0.18), depthWrite: false,
    }));
  water.position.set((FIELD.x0 + FIELD.x1) / 2 + 0.1, 0, (FIELD.z0 + FIELD.z1) / 2 + 0.05);
  water.renderOrder = 2;
  scene.add(water);

  // palms on the far side of the canal and along the left
  const rng = mulberry32(7);
  const palms = [];
  for (let i = 0; i < 16; i++) {
    const p = palm(rng);
    const left = i >= 11;
    p.position.set(left ? -12 - rng() * 14 : -20 + rng() * 44,
      0, left ? -3 + rng() * 6 : -8.5 - rng() * 14);
    scene.add(p);
    palms.push(p);
  }

  // clouds — darken when rain is falling or forecast
  const puffs = [];
  for (let i = 0; i < 6; i++) puffs.push(new THREE.SphereGeometry(1 + (i % 3) * 0.35, 12, 8)
    .translate(i * 1.1 - 2.8, Math.sin(i * 1.7) * 0.4, Math.cos(i * 2.3) * 0.7));
  const cloudGeo = mergeGeometries(puffs);
  const cloudMat = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 1, transparent: true, opacity: 0.92 });
  const clouds = [];
  for (let i = 0; i < 7; i++) {
    const c = new THREE.Mesh(cloudGeo, cloudMat);
    c.position.set(-30 + i * 10 + rng() * 4, 11 + rng() * 4, -24 + rng() * 20);
    c.scale.setScalar(1.2 + rng() * 0.9);
    scene.add(c);
    clouds.push(c);
  }

  // stratum + water labels on the front cut face
  const lblZ = PLOT.z1 + 0.02;
  const strata = [
    label('Puddled topsoil', 'plain', [0, 0.5]), label('Plough pan', 'plain', [0, 0.5]),
    label('Subsoil', 'plain', [0, 0.5]),
  ];
  strata[0].position.set(-2.4, mmY(-PUDDLED_MM / 2), lblZ);
  strata[1].position.set(-2.4, mmY(-(PUDDLED_MM + PAN_MM) / 2), lblZ);
  strata[2].position.set(-2.4, mmY(-330), lblZ);
  const wtLabel = label('', 'sky', [0, 1.15]);
  wtLabel.position.set(1.5, 0, lblZ);
  const pondLabel = label('', 'water', [0, -0.2]);
  pondLabel.position.set(1.5, 0, lblZ);
  for (const o of [...strata, wtLabel, pondLabel]) scene.add(o);

  const sky = { stormy: 0 };
  const cloudWhite = new THREE.Color('#ffffff'), cloudGrey = new THREE.Color('#6f7a80');

  return {
    shared, sun, palms,
    update(v, s, t, dt) {
      shared.uTime.value = t;
      if (s.bund_mm !== bundBuiltFor) buildBunds(s.bund_mm);

      // the field's real water (the simulator), not the sensor's reading
      const wl = v.wlA;
      shared.uWT.value = Math.min(mmY(wl), 0);
      shared.uFloor.value = mmY(v.floor);
      shared.uSensor.value = mmY(v.wlS);
      shared.uDry.value = clamp((-wl - 20) / 100, 0, 1);
      surface.roughness = 0.35 + 0.6 * shared.uDry.value;

      const depth = mmY(clamp(wl, 0, s.bund_mm + 6));
      water.visible = depth > 0.003;
      water.scale.y = Math.max(depth, 0.001);
      water.position.y = depth / 2 + 0.001;
      waterTex.offset.set(t * 0.012, t * 0.007);

      setLabel(wtLabel, `Water table ${Math.round(wl)} mm`, wl < -4);
      wtLabel.position.y = mmY(wl);
      setLabel(pondLabel, `Ponded ${Math.round(wl)} mm`, wl > 4);
      pondLabel.position.y = depth;

      // neighbours ripen with the plot
      const nc = keyframes([[0, [0.62, 0.78, 0.40]], [0.6, [0.48, 0.70, 0.32]],
        [0.85, [0.80, 0.76, 0.40]], [1, [0.86, 0.74, 0.42]]], v.g);
      for (const m of neighbours) m.color.setRGB(nc[0], nc[1], nc[2], THREE.SRGBColorSpace);

      // weather
      const stormy = clamp(Math.max(v.rain / 25, v.fcRain / 60, v.delay * 0.6), 0, 1);
      sky.stormy += (stormy - sky.stormy) * Math.min(dt * 2, 1);
      cloudMat.color.copy(cloudWhite).lerp(cloudGrey, sky.stormy);
      sun.intensity = 2.4 * (1 - 0.55 * sky.stormy);
      for (const c of clouds) {
        c.position.x += dt * 0.35;
        if (c.position.x > 40) c.position.x = -40;
      }
      return sky.stormy;
    },
  };
}
