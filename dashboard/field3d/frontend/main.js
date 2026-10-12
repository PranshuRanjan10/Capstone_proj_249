// Field 3D — entry point.
//
// Speaks the Streamlit component protocol directly (no streamlit-component-lib):
// the iframe announces itself with componentReady, then receives a "render"
// message with the scene state on every Streamlit rerun. The WebGL scene is
// built once; each new state only retargets values that are eased toward
// every frame, so moving the day slider animates water and crop growth.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { createWorld } from './scene.js';
import { createRice } from './rice.js';
import { createGauge } from './gauge.js';
import { createFx } from './fx.js';
import { loadOptionalModels } from './models.js';
import { ACTION_COLOUR, ACTION_GLOSS, C, VEX } from './config.js';

// ── Streamlit protocol ──────────────────────────────────────────────────────
const post = (type, data = {}) =>
  window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, '*');

let S = null;                       // latest state from Python (discrete values)
let frameHeight = 0;
let pending = null;                 // state that arrived before the scene was built
let models = null;                  // optional GLBs present on disk (from Python)
let onState = (s) => { pending = s; };
window.addEventListener('message', (e) => {
  if (!e.data || e.data.type !== 'streamlit:render') return;
  const args = e.data.args || {};
  const h = args.height || 620;
  if (h !== frameHeight) { frameHeight = h; post('streamlit:setFrameHeight', { height: h }); }
  if (args.models && !models) models = args.models;
  if (args.state) onState(args.state);
});
post('streamlit:componentReady', { apiVersion: 1 });

// ── renderer, camera, controls ──────────────────────────────────────────────
const stageEl = document.getElementById('stage');
const canvas = document.getElementById('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const labels = new CSS2DRenderer({ element: document.getElementById('labels') });

const scene = new THREE.Scene();
scene.background = new THREE.Color('#dfe9ee');
scene.fog = new THREE.Fog('#dfe9ee', 45, 150);

const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 400);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.minDistance = 3;
controls.maxDistance = 45;
controls.maxPolarAngle = Math.PI / 2 + 0.12;

const VIEWS = {
  overview: { pos: [14, 6.4, 17.5], target: [0.4, -0.2, 0.4] },
  cutaway: { pos: [2.2, 0.2, 13.5], target: [1.0, -0.6, 3.6] },
  sensor: { pos: [3.1, 0.55, 9.8], target: [2.5, -0.4, 4.0] },
};
let flight = null;
function flyTo(name, instant = false) {
  const v = VIEWS[name];
  const to = { pos: new THREE.Vector3(...v.pos), target: new THREE.Vector3(...v.target) };
  if (instant) { camera.position.copy(to.pos); controls.target.copy(to.target); return; }
  flight = { from: { pos: camera.position.clone(), target: controls.target.clone() }, to, t: 0 };
}
flyTo('overview', true);
document.querySelectorAll('.views button').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.views button').forEach((o) => o.classList.toggle('on', o === b));
  flyTo(b.dataset.view);
}));

// ── assets ──────────────────────────────────────────────────────────────────
const texLoader = new THREE.TextureLoader();
const loadTex = (name, srgb) => texLoader.loadAsync(`./assets/textures/${name}`).then((t) => {
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  return t;
});

const [wetDiff, wetNor, dryDiff, dryNor, grass] = await Promise.all([
  loadTex('mud_wet_diff.jpg', true), loadTex('mud_wet_nor.jpg', false),
  loadTex('mud_dry_diff.jpg', true), loadTex('mud_dry_nor.jpg', false),
  loadTex('grass_diff.jpg', true),
]);

new RGBELoader().load('./assets/hdri/sky_1k.hdr', (hdr) => {
  hdr.mapping = THREE.EquirectangularReflectionMapping;
  scene.environment = hdr;
  scene.environmentIntensity = 0.7;
  scene.background = hdr;
  scene.backgroundIntensity = 0.95;
});

const world = createWorld(scene, { wetDiff, wetNor, dryDiff, dryNor, grass });
const rice = createRice(scene);
const gauge = createGauge(scene);
const fx = createFx(scene);

// ── state → eased values ────────────────────────────────────────────────────
// T: targets from the latest state. V: what is drawn, eased toward T.
function targets(s) {
  const q = {};
  for (const h of [1, 2, 3]) {
    const f = s.forecast[String(h)] || { q05: s.wl_sensor_mm, q50: s.wl_sensor_mm, q95: s.wl_sensor_mm };
    q[h] = [f.q05, f.q50, f.q95];
  }
  return {
    wlS: s.wl_sensor_mm, wlA: s.wl_actual_mm, g: s.gdd_fraction, floor: s.floor_mm,
    bund: s.bund_mm, rain: s.rain_today_mm, fcRain: s.rain_fc_sum3_mm, q,
    pump: s.action === 'IRRIGATE' ? 1 : 0,
    drain: s.action === 'DRAIN' ? 1 : 0,
    inspect: s.action === 'INSPECT' ? 1 : 0,
    delay: s.action === 'DELAY' ? 1 : 0,
  };
}
let T = null, V = null;
const clone = (o) => JSON.parse(JSON.stringify(o));
function ease(cur, tgt, k) {
  for (const key in tgt) {
    if (typeof tgt[key] === 'number') cur[key] += (tgt[key] - cur[key]) * k;
    else if (tgt[key] && typeof tgt[key] === 'object') ease(cur[key], tgt[key], k);
  }
}

// ── HUD ─────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const sign = (mm) => `${mm > 0 ? '+' : mm < 0 ? '−' : ''}${Math.abs(Math.round(mm))} mm`;
const title = (s) => s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
$('h-vex').textContent = VEX;

function renderHud(s) {
  $('h-season').textContent = `Day ${s.das} · ${s.season}`;
  $('h-stage').textContent = title(s.stage);
  const tl = $('h-timeline');
  if (!tl.dataset.built) {
    tl.innerHTML = s.stages.map((st) => `<div class="seg" data-name="${st.name}" title="${title(st.name)}"
      style="flex:${(st.end - st.start).toFixed(3)}"></div>`).join('') + '<div class="mark"></div>';
    tl.dataset.built = '1';
  }
  tl.querySelectorAll('.seg').forEach((el) => {
    el.classList.toggle('on', el.dataset.name === s.stage);
    el.classList.toggle('nodrain', ['panicle_initiation', 'flowering'].includes(el.dataset.name));
  });
  tl.querySelector('.mark').style.left = `${(s.gdd_fraction * 100).toFixed(1)}%`;

  const pill = $('h-action');
  pill.textContent = s.action;
  pill.style.background = ACTION_COLOUR[s.action] || C.ink;
  $('h-gloss').textContent = s.constraint_applied
    ? 'Drain blocked by the hard agronomic constraint'
    : ACTION_GLOSS[s.action] || '';
  const span = Math.max(s.interval_width_mm * 1.25, s.gate_mm * 1.45);
  $('h-width').textContent = `${Math.round(s.interval_width_mm)} mm`;
  $('h-fill').style.width = `${Math.min(100, (s.interval_width_mm / span) * 100)}%`;
  $('h-fill').style.background = s.gated ? 'var(--rust)' : 'var(--rice)';
  $('h-gate').style.left = `${(s.gate_mm / span) * 100}%`;
  $('h-gate-txt').textContent = s.gated
    ? `exceeds the ${Math.round(s.gate_mm)} mm gate — automation refused`
    : `${Math.round(s.gate_mm - s.interval_width_mm)} mm margin to the ${Math.round(s.gate_mm)} mm gate`;

  const faulted = s.fault !== 'none';
  document.querySelector('.bl').classList.toggle('faulted', faulted);
  $('h-wl').textContent = faulted ? `${sign(s.wl_sensor_mm)} (sensor)` : sign(s.wl_sensor_mm);
  $('h-wl-act').textContent = sign(s.wl_actual_mm);
  $('h-rain').textContent = `${s.rain_today_mm.toFixed(1)} mm`;
  $('h-fc').textContent = `${s.rain_fc_sum3_mm.toFixed(0)} mm`;
  $('h-sm').textContent = `${s.sm_pct.toFixed(1)} %`;
}

onState = (s) => {
  S = s;
  T = targets(s);
  if (!V) V = clone(T);
  renderHud(s);
};
if (pending) onState(pending);

// optional GLBs: wait until Python has told us which exist, then load once
(function waitForModels() {
  if (!models) return setTimeout(waitForModels, 300);
  if (models.length) loadOptionalModels(models, scene, world, fx);
})();

// ── resize + visibility ─────────────────────────────────────────────────────
let W = 0, H = 0;
new ResizeObserver(() => {
  W = stageEl.clientWidth; H = stageEl.clientHeight;
  if (!W || !H) return;
  renderer.setSize(W, H, false);
  labels.setSize(W, H);
  camera.aspect = W / H;
  camera.updateProjectionMatrix();
}).observe(stageEl);

// ── loop ────────────────────────────────────────────────────────────────────
const clock = new THREE.Clock();
let shown = false;
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (!W || !H || document.hidden || !S) return;     // hidden tab or no data yet
  const t = clock.elapsedTime;

  ease(V, T, 1 - Math.exp(-dt * 4));
  if (flight) {
    flight.t = Math.min(flight.t + dt / 1.2, 1);
    const k = flight.t * flight.t * (3 - 2 * flight.t);
    camera.position.lerpVectors(flight.from.pos, flight.to.pos, k);
    controls.target.lerpVectors(flight.from.target, flight.to.target, k);
    if (flight.t >= 1) flight = null;
  }
  controls.update();

  world.update(V, S, t, dt);
  rice.update(V, t);
  gauge.update(V, S, t);
  fx.update(V, S, t, dt);

  renderer.render(scene, camera);
  labels.render(scene, camera);
  if (!shown) { shown = true; $('loading').classList.add('done'); }
});
