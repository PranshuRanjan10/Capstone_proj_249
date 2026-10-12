// Instruments on the front cut face: the AWD field-water tube (above-ground
// part — the buried part is painted by the strata shader), the ESP32 post,
// a mm ruler, the bund / AWD-floor reference lines, and the forecast fan
// (the dashboard's forecast chart, standing in the soil it describes).
import * as THREE from 'three';
import { label, setLabel } from './scene.js';
import {
  ACTION_COLOUR, BOTTOM, C, FAN, PLOT, POST, TUBE, WL_MIN_MM, mmY, clamp,
} from './config.js';

const FACE_Z = PLOT.z1;
const fmt = (mm) => `${mm > 0 ? '+' : mm < 0 ? '−' : ''}${Math.abs(Math.round(mm))}`;

function dashedLine(colour, x0, x1) {
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x0, 0, 0), new THREE.Vector3(x1, 0, 0)]);
  const l = new THREE.Line(g, new THREE.LineDashedMaterial({ color: colour, dashSize: 0.18, gapSize: 0.1 }));
  l.computeLineDistances();
  return l;
}

export function createGauge(scene) {
  const g = new THREE.Group();
  scene.add(g);

  // ── AWD tube above ground: back half only, as if sliced by the cut ────────
  const TUBE_UP = 0.28;
  const tube = new THREE.Mesh(
    new THREE.CylinderGeometry(TUBE.r, TUBE.r, TUBE_UP, 20, 1, true, Math.PI / 2, Math.PI),
    new THREE.MeshPhysicalMaterial({ color: '#e8ecea', roughness: 0.3, transparent: true, opacity: 0.75, side: THREE.DoubleSide }));
  tube.position.set(TUBE.x, TUBE_UP / 2, FACE_Z);
  g.add(tube);
  const tubeWater = new THREE.Mesh(
    new THREE.CylinderGeometry(TUBE.r * 0.94, TUBE.r * 0.94, 1, 20, 1, false, Math.PI / 2, Math.PI),
    new THREE.MeshStandardMaterial({ color: '#3c8cb4', transparent: true, opacity: 0.85 }));
  tubeWater.position.set(TUBE.x, 0, FACE_Z - 0.002);
  g.add(tubeWater);
  const float = new THREE.Mesh(new THREE.BoxGeometry(TUBE.r * 1.8, 0.014, 0.08),
    new THREE.MeshBasicMaterial({ color: '#f98f1f' }));
  float.position.set(TUBE.x, 0, FACE_Z - 0.045);
  g.add(float);

  // ── ESP32 post, LED, solar panel, cable to the tube, INSPECT beacon ──────
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.04, 1.5, 8),
    new THREE.MeshStandardMaterial({ color: '#8a7456', roughness: 0.9 }));
  post.position.set(POST.x, 0.75, POST.z);
  post.castShadow = true;
  g.add(post);
  const box = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.17, 0.12),
    new THREE.MeshStandardMaterial({ color: '#f2f4f1', roughness: 0.4 }));
  box.position.set(POST.x, 1.42, POST.z + 0.06);
  box.castShadow = true;
  g.add(box);
  const ledMat = new THREE.MeshStandardMaterial({ color: '#000', emissive: C.rice, emissiveIntensity: 2.5 });
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.022, 10, 8), ledMat);
  led.position.set(POST.x + 0.07, 1.44, POST.z + 0.125);
  g.add(led);
  const panel = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.015, 0.24),
    new THREE.MeshStandardMaterial({ color: '#1d2f4a', roughness: 0.25, metalness: 0.6 }));
  panel.position.set(POST.x, 1.6, POST.z);
  panel.rotation.x = -0.5;
  panel.castShadow = true;
  g.add(panel);
  const cable = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
    new THREE.Vector3(POST.x + 0.08, 1.34, POST.z + 0.08), new THREE.Vector3(POST.x + 0.35, 0.6, POST.z + 0.2),
    new THREE.Vector3(TUBE.x - 0.1, 0.4, FACE_Z - 0.06), new THREE.Vector3(TUBE.x, TUBE_UP + 0.02, FACE_Z - 0.04),
  ]), 20, 0.008, 5), new THREE.MeshStandardMaterial({ color: '#222' }));
  g.add(cable);

  const beaconMat = new THREE.MeshStandardMaterial({ color: '#400', emissive: '#ff2a12', emissiveIntensity: 0 });
  const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.055, 14, 10), beaconMat);
  beacon.position.set(POST.x, 1.56, POST.z + 0.06);
  g.add(beacon);
  const beaconLight = new THREE.PointLight('#ff3a20', 0, 6, 1.6);
  beaconLight.position.copy(beacon.position);
  g.add(beaconLight);

  const sensorLbl = label('', '', [0.5, 1.4]);
  sensorLbl.position.set(POST.x, 1.75, POST.z);
  const alertLbl = label('Automation blocked — inspect the field', 'alert', [0.5, 2.6]);
  alertLbl.position.set(POST.x, 1.75, POST.z);
  g.add(sensorLbl, alertLbl);

  // ── reference lines across the front face ────────────────────────────────
  const floorLine = dashedLine(C.rust, PLOT.x0, PLOT.x1);
  floorLine.position.z = FACE_Z + 0.012;
  const bundLine = dashedLine(C.sky, PLOT.x0, PLOT.x1);
  bundLine.position.z = FACE_Z + 0.012;
  g.add(floorLine, bundLine);
  const floorLbl = label('', 'rust', [0, -0.15]);
  const bundLbl = label('', 'sky', [0, -0.15]);
  floorLbl.position.set(PLOT.x0 + 0.25, 0, FACE_Z + 0.02);
  bundLbl.position.set(PLOT.x0 + 0.25, 0, FACE_Z + 0.02);
  g.add(floorLbl, bundLbl);

  // ── mm ruler at the front-right corner ──────────────────────────────────
  const RX = PLOT.x1 - 0.06, RZ = FACE_Z + 0.03;
  const white = new THREE.MeshBasicMaterial({ color: '#f5f5ef' });
  const top = mmY(150);
  const ruler = new THREE.Mesh(new THREE.BoxGeometry(0.03, top - BOTTOM, 0.02), white);
  ruler.position.set(RX, (top + BOTTOM) / 2, RZ);
  g.add(ruler);
  for (let mm = -350; mm <= 150; mm += 50) {
    const major = mm % 100 === 0;
    const tick = new THREE.Mesh(new THREE.BoxGeometry(major ? 0.14 : 0.08, 0.008, 0.02), white);
    tick.position.set(RX - (major ? 0.07 : 0.04), mmY(mm), RZ);
    g.add(tick);
    if (major) {
      const l = label(mm === 0 ? '0 mm' : fmt(mm), 'plain', [0, 0.5]);
      l.position.set(RX + 0.06, mmY(mm), RZ);
      g.add(l);
    }
  }

  // ── forecast fan: today + 1/2/3 days, q05–q95 bar and q50 dot ────────────
  const barMat = new THREE.MeshBasicMaterial({ color: C.fan, transparent: true, opacity: 0.55, depthWrite: false });
  const cols = [];
  for (let i = 0; i < 4; i++) {
    const x = FAN.x0 + i * FAN.dx;
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.13, 1, 0.02), barMat);
    bar.position.set(x, 0, FAN.z);
    bar.renderOrder = 3;
    bar.visible = i > 0;
    const dot = new THREE.Mesh(new THREE.SphereGeometry(i ? 0.045 : 0.055, 14, 10),
      new THREE.MeshBasicMaterial({ color: i ? '#ffffff' : '#f98f1f' }));
    dot.position.set(x, 0, FAN.z + 0.02);
    const val = label('', 'water', [0.5, -0.5]);
    val.position.set(x, 0, FAN.z);
    const day = label(i ? `+${i}d` : 'today', 'plain', [0.5, 0]);
    day.position.set(x, BOTTOM + 0.04, FAN.z);
    g.add(bar, dot, val, day);
    cols.push({ bar, dot, val });
  }
  const linePos = new Float32Array(12);
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.BufferAttribute(linePos, 3));
  const fanLine = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: '#ffffff' }));
  g.add(fanLine);
  const fanTitle = label('Forecast · 90% interval', 'water', [0.5, 1]);
  fanTitle.position.set(FAN.x0 + 1.5 * FAN.dx, BOTTOM - 0.22, FAN.z);
  g.add(fanTitle);

  const ledColour = new THREE.Color();
  const shownY = (mm) => mmY(clamp(mm, WL_MIN_MM - 90, 150));

  return {
    update(v, s, t) {
      // sensor reading (may be faulted) inside the tube
      const ys = mmY(v.wlS);
      tubeWater.visible = v.wlS > 1;
      tubeWater.scale.y = Math.max(Math.min(ys, TUBE_UP), 0.001);
      tubeWater.position.y = tubeWater.scale.y / 2;
      float.position.y = clamp(ys, mmY(-300), TUBE_UP);
      float.visible = ys > 0;                       // below ground the shader draws it

      const faulted = s.fault !== 'none';
      const flick = faulted ? (Math.sin(t * 23) > 0.2 ? 1 : 0.15) : 1;
      ledMat.emissive.copy(ledColour.set(ACTION_COLOUR[s.action] || C.ink));
      ledMat.emissiveIntensity = 2.5 * flick;
      setLabel(sensorLbl, faulted
        ? `⚠ ${s.fault} · reads ${fmt(v.wlS)} mm`
        : `ESP32 · ${fmt(v.wlS)} mm`);
      sensorLbl.el.className = `lbl3d ${faulted ? 'rust' : ''}`;

      const k = v.inspect;
      const pulse = 0.5 + 0.5 * Math.sin(t * 7);
      beaconMat.emissiveIntensity = k * (1.5 + 3 * pulse);
      beaconLight.intensity = k * (2 + 6 * pulse);
      setLabel(alertLbl, alertLbl.el.textContent, s.action === 'INSPECT');

      floorLine.position.y = mmY(v.floor);
      setLabel(floorLbl, `AWD floor ${fmt(v.floor)} mm · ${s.stage.replace(/_/g, ' ')}`);
      floorLbl.position.y = mmY(v.floor);
      bundLine.position.y = mmY(v.bund);
      setLabel(bundLbl, `Bund overflow ${fmt(v.bund)} mm`);
      bundLbl.position.y = mmY(v.bund);

      // forecast fan
      const pts = [[v.wlS, v.wlS, v.wlS], ...[1, 2, 3].map((h) => v.q[h])];
      pts.forEach(([lo, mid, hi], i) => {
        const c = cols[i];
        const y0 = shownY(lo), y1 = shownY(hi), ym = shownY(mid);
        c.bar.scale.y = Math.max(y1 - y0, 0.004);
        c.bar.position.y = (y0 + y1) / 2;
        c.dot.position.y = ym;
        setLabel(c.val, `${fmt(mid)}`, i > 0);
        c.val.position.y = y1;
        linePos.set([c.dot.position.x, ym, FAN.z + 0.02], i * 3);
      });
      lineGeo.attributes.position.needsUpdate = true;
    },
  };
}
