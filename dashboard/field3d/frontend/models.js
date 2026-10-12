// Optional glTF models. The scene is complete without them; when a file is
// dropped into assets/models/ the Python wrapper lists it in `models`, and it
// replaces (palm, pump) or adds to (farmer) the procedural geometry.
//
//   farmer.glb  a person, for scale — stands on the back bund
//   palm.glb    coconut palm — replaces every procedural palm
//   pump.glb    diesel pump set — replaces the procedural pump body
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PLOT, BUND_W, mmY } from './config.js';

// Scale a loaded scene so its largest (or vertical) extent is `size` metres,
// centre it on x/z and sit it on y = 0.
function normalise(root, size, byHeight = false) {
  const box = new THREE.Box3().setFromObject(root);
  const dim = box.getSize(new THREE.Vector3());
  const s = size / (byHeight ? dim.y : Math.max(dim.x, dim.y, dim.z));
  root.scale.setScalar(s);
  const b2 = new THREE.Box3().setFromObject(root);
  const c = b2.getCenter(new THREE.Vector3());
  root.position.set(-c.x, -b2.min.y, -c.z);
  const holder = new THREE.Group();
  holder.add(root);
  root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return holder;
}

export function loadOptionalModels(names, scene, world, fx) {
  const have = new Set(names || []);
  const loader = new GLTFLoader();
  const load = (n) => loader.loadAsync(`./assets/models/${n}`).catch((e) => {
    console.warn(`Field 3D: could not load optional model ${n}`, e);
    return null;
  });

  if (have.has('farmer.glb')) {
    load('farmer.glb').then((g) => {
      if (!g) return;
      const f = normalise(g.scene, 1.65, true);
      f.position.set(PLOT.x0 + 3.2, mmY(100) + 0.1, PLOT.z0 + BUND_W / 2);
      f.rotation.y = Math.PI * 0.15;
      scene.add(f);
      if (g.animations.length) {           // play an idle clip if the model has one
        const mixer = new THREE.AnimationMixer(g.scene);
        const clip = g.animations.find((a) => /idle/i.test(a.name)) || g.animations[0];
        mixer.clipAction(clip).play();
        const clock = new THREE.Clock();
        (function tick() { mixer.update(clock.getDelta()); requestAnimationFrame(tick); })();
      }
    });
  }
  if (have.has('palm.glb')) {
    load('palm.glb').then((g) => {
      if (!g) return;
      const proto = normalise(g.scene, 8, true);
      for (const p of world.palms) {
        const c = proto.clone(true);
        c.position.copy(p.position);
        c.rotation.y = p.position.x * 1.7;
        c.scale.setScalar(0.8 + ((p.position.z * 13) % 1 + 1) % 1 * 0.4);
        scene.add(c);
        p.visible = false;
      }
    });
  }
  if (have.has('pump.glb')) {
    load('pump.glb').then((g) => {
      if (!g) return;
      const m = normalise(g.scene, 0.9);
      for (const c of fx.pump.children) c.visible = false;
      fx.pump.add(m);
    });
  }
}
