// The decision, acted out: pump + inflow for IRRIGATE, weir + outflow for
// DRAIN, a padlock on the weir while drainage is agronomically forbidden, and
// rain. Particle systems are small and CPU-driven; only active particles are
// drawn.
import * as THREE from 'three';
import { label, setLabel } from './scene.js';
import { BUND_W, C, PLOT, PUMP, WEIR, mmY, clamp } from './config.js';

function particles(n, colour, size) {
  const pos = new Float32Array(n * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setDrawRange(0, 0);
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({
    color: colour, size, transparent: true, opacity: 0.85, depthWrite: false,
  }));
  pts.frustumCulled = false;
  return { pts, pos, vel: new Float32Array(n * 3), n };
}

export function createFx(scene) {
  // ── diesel pump on the far bank, lifting from the canal ──────────────────
  const pump = new THREE.Group();
  pump.position.set(PUMP.x, 0, PUMP.z);
  scene.add(pump);
  const green = new THREE.MeshStandardMaterial({ color: '#2f6b3a', roughness: 0.5, metalness: 0.3 });
  const steel = new THREE.MeshStandardMaterial({ color: '#4a4f52', roughness: 0.4, metalness: 0.7 });
  const skid = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.08, 0.55), steel);
  skid.position.y = 0.04;
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.36, 0.4), green);
  body.position.y = 0.28;
  const head = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.13, 0.22, 12), steel);
  head.position.set(-0.12, 0.56, 0);
  const fly = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.05, 20), steel);
  fly.rotation.z = Math.PI / 2;
  fly.position.set(-0.36, 0.3, 0);
  const spoke = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.36, 0.03), green);
  fly.add(spoke);
  spoke.rotation.z = Math.PI / 2;
  const exhaust = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.4, 8), steel);
  exhaust.position.set(0.18, 0.62, -0.1);
  for (const m of [skid, body, head, fly, exhaust]) { m.castShadow = true; pump.add(m); }

  const pipeMat = new THREE.MeshStandardMaterial({ color: '#2d5a8a', roughness: 0.5 });
  const px = PUMP.x + 0.25;
  const suction = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
    new THREE.Vector3(px - 0.1, 0.25, PUMP.z + 0.2), new THREE.Vector3(px - 0.1, 0.25, -5.55),
    new THREE.Vector3(px - 0.1, -0.3, -5.2),
  ]), 16, 0.045, 8), pipeMat);
  const outlet = new THREE.Vector3(px, 0.42, PLOT.z0 + BUND_W + 0.25);
  const delivery = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
    new THREE.Vector3(px, 0.35, PUMP.z + 0.2), new THREE.Vector3(px, 0.85, -5.1),
    new THREE.Vector3(px, 0.75, PLOT.z0 + 0.1), outlet,
  ]), 32, 0.055, 8), pipeMat);
  for (const m of [suction, delivery]) { m.castShadow = true; scene.add(m); }
  const pumpLbl = label('Pump running', 'sky', [0.5, 1.6]);
  pumpLbl.position.set(PUMP.x, 0.8, PUMP.z);
  scene.add(pumpLbl);

  const inflow = particles(260, '#bfe6ff', 0.07);
  scene.add(inflow.pts);
  const smoke = particles(40, '#8b8f90', 0.22);
  smoke.pts.material.opacity = 0.45;
  scene.add(smoke.pts);

  // ── drainage weir in the left bund, with a padlock ───────────────────────
  const weir = new THREE.Group();
  weir.position.set(PLOT.x0 + BUND_W / 2, 0, WEIR.z);
  scene.add(weir);
  const wood = new THREE.MeshStandardMaterial({ color: '#7a5a3a', roughness: 0.9 });
  for (const dz of [-WEIR.w / 2, WEIR.w / 2]) {
    const p = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.75, 0.1), wood);
    p.position.set(0, 0.25, dz);
    p.castShadow = true;
    weir.add(p);
  }
  const plankH = mmY(110) + 0.1;
  const plank = new THREE.Mesh(new THREE.BoxGeometry(0.06, plankH, WEIR.w - 0.08), wood);
  plank.castShadow = true;
  weir.add(plank);
  const sill = new THREE.Mesh(new THREE.BoxGeometry(BUND_W, 0.04, WEIR.w - 0.1), wood);
  sill.position.y = 0.02;
  weir.add(sill);

  const lock = new THREE.Group();
  const lockMat = new THREE.MeshStandardMaterial({ color: C.rust, metalness: 0.6, roughness: 0.3, emissive: C.rust, emissiveIntensity: 0 });
  const lockBody = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.13, 0.16), lockMat);
  const shackle = new THREE.Mesh(new THREE.TorusGeometry(0.055, 0.014, 8, 16, Math.PI), lockMat);
  shackle.rotation.y = Math.PI / 2;
  shackle.position.y = 0.065;
  lock.add(lockBody, shackle);
  lock.position.set(0.06, plankH + 0.1, 0);
  weir.add(lock);
  const lockLbl = label('Drainage locked · panicle initiation → flowering', 'rust', [1, 1.5]);
  lockLbl.position.set(PLOT.x0, plankH + 0.25, WEIR.z);
  const drainLbl = label('Draining', 'water', [1, 1.5]);
  drainLbl.position.set(PLOT.x0, plankH + 0.25, WEIR.z);
  scene.add(lockLbl, drainLbl);

  const outflow = particles(220, '#9fd2dc', 0.07);
  scene.add(outflow.pts);

  // ── rain streaks ─────────────────────────────────────────────────────────
  const RAIN = 2400;
  const rainPos = new Float32Array(RAIN * 6);
  for (let i = 0; i < RAIN; i++) {
    const x = (Math.random() - 0.5) * 30, y = Math.random() * 12, z = (Math.random() - 0.5) * 24;
    rainPos.set([x, y, z, x + 0.02, y + 0.28, z], i * 6);
  }
  const rainGeo = new THREE.BufferGeometry();
  rainGeo.setAttribute('position', new THREE.BufferAttribute(rainPos, 3));
  rainGeo.setDrawRange(0, 0);
  const rain = new THREE.LineSegments(rainGeo, new THREE.LineBasicMaterial({ color: '#c9dbe6', transparent: true, opacity: 0.55 }));
  rain.frustumCulled = false;
  scene.add(rain);

  let nIn = 0, nOut = 0, nSmoke = 0;

  function spawnIn(i) {
    inflow.pos.set([outlet.x + (Math.random() - 0.5) * 0.06, outlet.y, outlet.z + 0.05], i * 3);
    inflow.vel.set([(Math.random() - 0.5) * 0.25, -0.1 + Math.random() * 0.2, 1.1 + Math.random() * 0.4], i * 3);
  }
  function spawnOut(i, y) {
    outflow.pos.set([PLOT.x0 + BUND_W - 0.05, y, WEIR.z + (Math.random() - 0.5) * (WEIR.w - 0.2)], i * 3);
    outflow.vel.set([-1.1 - Math.random() * 0.5, 0.05, (Math.random() - 0.5) * 0.2], i * 3);
  }
  function spawnSmoke(i) {
    smoke.pos.set([PUMP.x + 0.18, 0.85, PUMP.z - 0.1], i * 3);
    smoke.vel.set([(Math.random() - 0.5) * 0.1, 0.5 + Math.random() * 0.3, 0.1], i * 3);
  }
  // advance a particle system; respawn when `dead(x, y, z)` is true
  function step(p, n, dt, g, dead, spawn) {
    for (let i = 0; i < n; i++) {
      const k = i * 3;
      p.vel[k + 1] -= g * dt;
      p.pos[k] += p.vel[k] * dt; p.pos[k + 1] += p.vel[k + 1] * dt; p.pos[k + 2] += p.vel[k + 2] * dt;
      if (dead(p.pos[k], p.pos[k + 1], p.pos[k + 2])) spawn(i);
    }
    p.pts.geometry.setDrawRange(0, n);
    p.pts.geometry.attributes.position.needsUpdate = true;
  }

  return {
    pump,
    update(v, s, t, dt) {
      dt = Math.min(dt, 0.05);
      const surfaceY = Math.max(mmY(v.wlA), 0);

      // pump
      const on = v.pump;
      fly.rotation.x += dt * 22 * on;
      body.position.x = on > 0.1 ? Math.sin(t * 60) * 0.004 : 0;
      setLabel(pumpLbl, 'Pump running · irrigating', on > 0.5);
      const wantIn = Math.round(inflow.n * on);
      while (nIn < wantIn) spawnIn(nIn++);
      nIn = Math.min(nIn, wantIn);
      step(inflow, nIn, dt, 3.5, (x, y, z) => y < surfaceY + 0.01 || z > outlet.z + 2, spawnIn);
      const wantSmoke = Math.round(smoke.n * on);
      while (nSmoke < wantSmoke) spawnSmoke(nSmoke++);
      nSmoke = Math.min(nSmoke, wantSmoke);
      step(smoke, nSmoke, dt, -0.05, (x, y) => y > 2.6, spawnSmoke);

      // weir + outflow
      const open = v.drain;
      plank.position.y = plankH / 2 + open * (plankH * 0.85);
      const outY = surfaceY > 0.01 ? surfaceY : 0;
      const wantOut = outY > 0.01 ? Math.round(outflow.n * open) : 0;
      while (nOut < wantOut) spawnOut(nOut++, outY);
      nOut = Math.min(nOut, wantOut);
      step(outflow, nOut, dt, 3, (x, y) => x < -7.2 || y < -0.2, (i) => spawnOut(i, outY));
      setLabel(drainLbl, 'Weir open · draining', open > 0.5);

      // padlock: present whenever the stage forbids drainage; glows when it
      // actually blocked a drain recommendation
      lock.visible = s.no_drain;
      lockMat.emissiveIntensity = s.constraint_applied ? 0.6 + 0.6 * Math.sin(t * 5) : 0.05;
      lock.position.y = plankH + 0.1 + open * plankH * 0.85;
      setLabel(lockLbl, s.constraint_applied
        ? 'DRAIN blocked · hard agronomic constraint'
        : 'Drainage locked · panicle initiation → flowering', s.no_drain);

      // rain: draw count follows today's rain, streaks wrap vertically
      const nRain = Math.round(RAIN * clamp(v.rain / 30, 0, 1));
      for (let i = 0; i < nRain; i++) {
        const k = i * 6;
        let y = rainPos[k + 1] - dt * 9;
        if (y < 0) y += 12;
        rainPos[k + 1] = y; rainPos[k + 4] = y + 0.28;
      }
      rainGeo.setDrawRange(0, nRain * 2);
      rainGeo.attributes.position.needsUpdate = nRain > 0;
    },
  };
}
