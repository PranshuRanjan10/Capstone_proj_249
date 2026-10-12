// Rice hills as two InstancedMeshes (leaf clumps + panicles). Every plant in
// the plot is at the same growth stage, so stage morphing is done with
// uniforms in the vertex shader — nothing is rebuilt as the season advances.
// Per-instance variety (height, rotation, tint) comes from a seed attribute.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { FIELD, PLOT, POST, TUBE, FAN, keyframes } from './config.js';

// Stage morphology over the GDD fraction (0 = transplanting, 1 = harvest).
const HEIGHT = [[0, 0.30], [0.15, 0.45], [0.35, 0.72], [0.5, 0.92], [0.62, 1.05], [1, 1.0]];
const SPREAD = [[0, 0.55], [0.15, 0.9], [0.35, 1.45], [0.5, 1.6], [1, 1.6]];
const LEAF_DROOP = [[0.75, 0], [1, 1]];
const BASE = [[0, [0.45, 0.62, 0.25]], [0.2, [0.22, 0.45, 0.15]], [0.6, [0.18, 0.40, 0.12]],
  [0.75, [0.36, 0.48, 0.16]], [0.9, [0.62, 0.56, 0.22]], [1, [0.70, 0.58, 0.26]]];
const TIP = [[0, [0.62, 0.78, 0.35]], [0.2, [0.40, 0.66, 0.25]], [0.6, [0.36, 0.62, 0.22]],
  [0.75, [0.62, 0.68, 0.28]], [0.9, [0.86, 0.76, 0.36]], [1, [0.90, 0.78, 0.40]]];
const PANICLE = [[0.55, 0], [0.66, 1]];
const PAN_DROOP = [[0.68, 0], [0.92, 1]];
const PAN_COL = [[0.62, [0.62, 0.70, 0.30]], [0.8, [0.80, 0.74, 0.32]], [0.92, [0.88, 0.72, 0.30]]];

// Shared GLSL: seed-driven rotation + height, and wind sway in world x.
const COMMON = /* glsl */`
  attribute float aSeed;
  uniform float uTime, uHeight, uSpread, uDroop, uWind;
  varying float vH; varying float vSeed;
  vec3 plantSpace(vec3 p, float h) {
    float a = aSeed * 6.2831;
    p.xz = mat2(cos(a), -sin(a), sin(a), cos(a)) * p.xz;
    float ph = instanceMatrix[3].x * 0.8 + instanceMatrix[3].z * 0.5;
    float w = 0.5 + 0.5 * sin(uTime * 1.3 + ph) * 0.7 + 0.3 * sin(uTime * 2.3 + ph * 1.9);
    p.x += (0.03 + 0.05 * w) * uWind * h * h * uHeight;
    return p;
  }`;

function instancedMaterial(vertexBody, uniforms) {
  const m = new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, roughness: 0.75 });
  m.customProgramCacheKey = () => vertexBody;      // leaf and panicle shaders differ
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = COMMON + '\n' + sh.vertexShader.replace('#include <begin_vertex>', vertexBody);
    sh.fragmentShader = 'uniform vec3 uBase, uTip; varying float vH; varying float vSeed;\n' +
      sh.fragmentShader.replace('#include <color_fragment>', `#include <color_fragment>
        diffuseColor.rgb *= mix(uBase, uTip, vH) * (0.86 + 0.28 * vSeed) * mix(0.55, 1.0, vH);`);
  };
  return m;
}

// One clump: ~10 tapered, outward-curving blades. Base at y = 0, unit height.
function clumpGeometry() {
  const blades = [];
  const N = 11;
  for (let i = 0; i < N; i++) {
    const g = new THREE.PlaneGeometry(0.024, 1, 1, 5);
    g.translate(0, 0.5, 0);
    const p = g.attributes.position;
    const hScale = 0.72 + ((i * 37) % 11) / 11 * 0.33;
    for (let k = 0; k < p.count; k++) {
      const y = p.getY(k);
      p.setX(k, p.getX(k) * (1 - y * 0.8));
      p.setY(k, y * hScale);
      p.setZ(k, 0.22 * y * y);                       // curve outward
    }
    g.rotateX(-0.12 - (i % 3) * 0.08);
    g.rotateY((i / N) * Math.PI * 2 + i * 0.4);
    blades.push(g);
  }
  const m = mergeGeometries(blades);
  m.computeVertexNormals();
  return m;
}

// Three arching panicle strands, anchored at the origin.
function panicleGeometry() {
  const strands = [];
  for (let i = 0; i < 3; i++) {
    const c = new THREE.QuadraticBezierCurve3(new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0.03, 0.13, 0), new THREE.Vector3(0.1, 0.2, 0));
    const g = new THREE.TubeGeometry(c, 6, 0.011, 4);
    g.rotateY((i / 3) * Math.PI * 2 + 0.5);
    strands.push(g);
  }
  return mergeGeometries(strands);
}

function hillPositions() {
  const pts = [];
  let s = 11;
  const r = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  for (let x = FIELD.x0 + 0.3; x < FIELD.x1 - 0.08; x += 0.3) {
    for (let z = FIELD.z0 + 0.3; z < FIELD.z1 - 0.08; z += 0.3) {
      const px = x + (r() - 0.5) * 0.06, pz = z + (r() - 0.5) * 0.06;
      // leave room for the instruments on the front cut face
      if (Math.hypot(px - POST.x, pz - POST.z) < 0.35) continue;
      if (Math.abs(px - TUBE.x) < 0.3 && pz > PLOT.z1 - 0.35) continue;
      if (px > FAN.x0 - 0.25 && pz > PLOT.z1 - 0.3) continue;
      pts.push([px, pz, r()]);
    }
  }
  return pts;
}

export function createRice(scene) {
  const pts = hillPositions();
  const seeds = new THREE.InstancedBufferAttribute(new Float32Array(pts.map((p) => p[2])), 1);

  const U = {
    uTime: { value: 0 }, uHeight: { value: 0.3 }, uSpread: { value: 0.6 },
    uDroop: { value: 0 }, uWind: { value: 1 },
  };
  const leafU = { ...U, uBase: { value: new THREE.Color() }, uTip: { value: new THREE.Color() } };
  const panU = {
    ...U, uPan: { value: 0 }, uPanDroop: { value: 0 },
    uBase: { value: new THREE.Color() }, uTip: { value: new THREE.Color() },
  };

  const leafMat = instancedMaterial(/* glsl */`
    float h = uv.y;
    vec3 transformed = position;
    transformed.xz *= mix(1.0, uSpread, h);
    transformed.y *= uHeight * (0.85 + 0.3 * aSeed);
    transformed.y -= uDroop * 0.18 * h * h * uHeight;
    transformed.xz *= 1.0 + uDroop * 0.25 * h;
    transformed = plantSpace(transformed, h);
    vH = h; vSeed = aSeed;`, leafU);

  const panMat = instancedMaterial(/* glsl */`
    float h = 1.0;
    vec3 transformed = position * uPan;
    float r = length(transformed.xz);
    transformed.y -= uPanDroop * r * 1.6;
    transformed.y += uHeight * (0.85 + 0.3 * aSeed) * (0.9 - 0.12 * uPanDroop);
    transformed = plantSpace(transformed, h);
    vH = clamp(position.y * 5.0, 0.0, 1.0); vSeed = aSeed;`, panU);
  panMat.onBeforeCompile = ((orig) => (sh) => {
    orig(sh);
    sh.vertexShader = 'uniform float uPan, uPanDroop;\n' + sh.vertexShader;
  })(panMat.onBeforeCompile);

  const make = (geo, mat) => {
    const m = new THREE.InstancedMesh(geo, mat, pts.length);
    geo.setAttribute('aSeed', seeds);
    const M = new THREE.Matrix4();
    pts.forEach(([x, z], i) => m.setMatrixAt(i, M.makeTranslation(x, 0, z)));
    m.frustumCulled = false;
    m.receiveShadow = true;
    scene.add(m);
    return m;
  };
  make(clumpGeometry(), leafMat);
  const panicles = make(panicleGeometry(), panMat);

  const setCol = (c, rgb) => c.setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);

  return {
    count: pts.length,
    update(v, t) {
      const g = v.g;
      U.uTime.value = t;
      U.uHeight.value = keyframes(HEIGHT, g);
      U.uSpread.value = keyframes(SPREAD, g);
      U.uDroop.value = keyframes(LEAF_DROOP, g);
      U.uWind.value = 1 + Math.min(v.rain / 20, 1.5);
      setCol(leafU.uBase.value, keyframes(BASE, g));
      setCol(leafU.uTip.value, keyframes(TIP, g));
      const pan = keyframes(PANICLE, g);
      panU.uPan.value = pan;
      panU.uPanDroop.value = keyframes(PAN_DROOP, g);
      const pc = keyframes(PAN_COL, g);
      setCol(panU.uBase.value, pc.map((c) => c * 0.8));
      setCol(panU.uTip.value, pc);
      panicles.visible = pan > 0.01;
    },
  };
}
