/* 3D embedding visualizer — continuous surface ("hill landscape") view.
   Lazy-loaded on scroll into view, separately from the dot-cloud view.
   Renders the same 67,736 books as a continuous sheet: a grid over the
   UMAP (x,z) plane where each cell's HEIGHT = book density (dense genre
   regions rise into hills) and COLOR = dominant genre in that cell. A
   "Flat sheet" toggle flattens the height for a pure paper-sheet look.

   Switch to this view with the Dots / Surface button in the UI. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DATA_URL = "static/embedding_3d.json";
const GRID = 128;                 // surface resolution (GRID x GRID cells)
const HEIGHT_SCALE = 0.5;         // peak elevation in the [-1,1] space
const BASE_HEIGHT = 0.08;         // lift the entire sheet so base isn't at y=0
const BLUR_PASSES = 2;            // smoothing passes for a continuous sheet
const DOMAIN_EXTEND = 0.3;        // extend domain beyond [-1,1] so edge hills have room
const BG = [0.043, 0.055, 0.078]; // matches --bg (#0b0e14) for empty cells

let state = null;
let renderer = null, scene = null, camera = null, controls = null;
let surfaceMesh = null, animating = false;
let heights = null;               // Float32Array(GRID*GRID) base hill heights
let cellGenre = null, cellCount = null, cellSamples = null;
let mode = "hills";
let resizeObserver = null;
let initialized = false, ready = false;

async function loadData() {
  const res = await fetch(DATA_URL);
  if (!res.ok) throw new Error("embedding_3d.json fetch failed: " + res.status);
  return res.json();
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255,
  ];
}

/* Bin every book into the grid: count, genre histogram, sample titles. */
function buildGrid(data) {
  const G = GRID;
  const nG = data.genres.length;
  const count = new Float32Array(G * G);
  const hist = Array.from({ length: G * G }, () => new Float32Array(nG));
  const samples = Array.from({ length: G * G }, () => []);
  for (let p = 0; p < data.points.length; p++) {
    const pt = data.points[p];
    const i = Math.min(G - 1, Math.max(0, Math.round(((pt.x + 1 + DOMAIN_EXTEND) / (2 + 2 * DOMAIN_EXTEND)) * (G - 1))));
    const j = Math.min(G - 1, Math.max(0, Math.round(((pt.z + 1 + DOMAIN_EXTEND) / (2 + 2 * DOMAIN_EXTEND)) * (G - 1))));
    const ci = j * G + i;
    count[ci] += 1;
    hist[ci][pt.label] += 1;
    if (samples[ci].length < 6) samples[ci].push(pt);
  }
  return { count, hist, samples, G, nG };
}

function blur(grid, G) {
  const out = new Float32Array(G * G);
  const w = [[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]];
  for (let j = 0; j < G; j++) {
    for (let i = 0; i < G; i++) {
      let s = 0, wsum = 0;
      for (let dj = -1; dj <= 1; dj++) {
        for (let di = -1; di <= 1; di++) {
          const ni = i + di, nj = j + dj;
          if (ni < 0 || nj < 0 || ni >= G || nj >= G) continue;
          const wt = w[dj + 1][di + 1];
          s += grid[nj * G + ni] * wt; wsum += wt;
        }
      }
      out[j * G + i] = s / wsum;
    }
  }
  return out;
}

function init() {
  const canvas = document.getElementById("embedding-3d-surface");
  const wrap = document.getElementById("embedding-3d-wrap");
  const W = wrap.clientWidth || 800, H = 560;
  canvas.width = W; canvas.height = H;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(W, H, false);
  renderer.setClearColor(0x0b0e14, 1);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(55, W / H, 0.01, 100);
  camera.position.set(1.3, 1.5, 1.7);
  camera.lookAt(0, 0.1, 0);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.1, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.rotateSpeed = 0.55;
  controls.zoomSpeed = 0.7;
  controls.panSpeed = 0.6;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.35;
  controls.update();

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.95);
  dir.position.set(0.8, 1.6, 0.5);
  scene.add(dir);
  const fill = new THREE.DirectionalLight(0x88aaff, 0.25);
  fill.position.set(-0.6, 0.4, -0.8);
  scene.add(fill);

  renderer.domElement.addEventListener("pointermove", onPointerMove, false);
  renderer.domElement.addEventListener("pointerdown", () => { controls.autoRotate = false; });
  renderer.domElement.addEventListener("wheel", () => { controls.autoRotate = false; }, { passive: true });

  resizeObserver = new ResizeObserver(() => {
    const w = wrap.clientWidth;
    if (w === 0 || !renderer) return;
    camera.aspect = w / H;
    camera.updateProjectionMatrix();
    renderer.setSize(w, H, false);
  });
  resizeObserver.observe(wrap);
}

function buildSurface(data) {
  const G = GRID;
  const { count, hist, samples } = buildGrid(data);
  cellSamples = samples;

  // Smooth the density field so the sheet is continuous, not blocky.
  let d = count;
  for (let k = 0; k < BLUR_PASSES; k++) d = blur(d, G);
  let maxD = 0;
  for (let i = 0; i < d.length; i++) maxD = Math.max(maxD, d[i]);
  const logMax = Math.log(1 + maxD) || 1;

  heights = new Float32Array(G * G);
  cellGenre = new Int16Array(G * G);
  cellCount = new Float32Array(G * G);
  const positions = new Float32Array(G * G * 3);
  const colors = new Float32Array(G * G * 3);

  for (let j = 0; j < G; j++) {
    for (let i = 0; i < G; i++) {
      const ci = j * G + i;
      const dens = d[ci];
      // Height = log density -> hills. Normalized to [0,1] then scaled.
      const hN = Math.log(1 + dens) / logMax;
      heights[ci] = hN * HEIGHT_SCALE;

      // Dominant genre for this cell.
      let best = -1, bestV = -1;
      const hh = hist[ci];
      for (let g = 0; g < hh.length; g++) {
        if (hh[g] > bestV) { bestV = hh[g]; best = g; }
      }
      cellGenre[ci] = best;
      cellCount[ci] = count[ci];

      const x = -1 - DOMAIN_EXTEND + (2 + 2 * DOMAIN_EXTEND) * i / (G - 1);
      const z = -1 - DOMAIN_EXTEND + (2 + 2 * DOMAIN_EXTEND) * j / (G - 1);
      positions[ci * 3] = x;
      positions[ci * 3 + 1] = heights[ci];
      positions[ci * 3 + 2] = z;

      // Color: blend genre color in by density so empty cells stay dark.
      const t = Math.min(1, Math.sqrt(hN));
      let r, g, b;
      if (best >= 0) {
        const c = hexToRgb(data.genres[best].color);
        r = BG[0] + (c[0] - BG[0]) * t;
        g = BG[1] + (c[1] - BG[1]) * t;
        b = BG[2] + (c[2] - BG[2]) * t;
      } else {
        r = BG[0]; g = BG[1]; b = BG[2];
      }
      colors[ci * 3] = r;
      colors[ci * 3 + 1] = g;
      colors[ci * 3 + 2] = b;
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const idx = [];
  for (let j = 0; j < G - 1; j++) {
    for (let i = 0; i < G - 1; i++) {
      const a = j * G + i, b = j * G + i + 1, c = (j + 1) * G + i, d2 = (j + 1) * G + i + 1;
      idx.push(a, c, b, b, c, d2);
    }
  }
  geom.setIndex(idx);
  geom.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.95, metalness: 0.0,
    side: THREE.DoubleSide, flatShading: false,
  });
  surfaceMesh = new THREE.Mesh(geom, mat);
  scene.add(surfaceMesh);
  applyMode(); // honor a pre-selected Flat/Hills mode
}

function applyMode() {
  if (!surfaceMesh) return;
  const pos = surfaceMesh.geometry.attributes.position;
  for (let ci = 0; ci < heights.length; ci++) {
    pos.array[ci * 3 + 1] = mode === "hills" ? heights[ci] : 0;
  }
  pos.needsUpdate = true;
  surfaceMesh.geometry.computeVertexNormals();
  surfaceMesh.geometry.computeBoundingSphere();
}

function onPointerMove(e) {
  if (!surfaceMesh) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1
  );
  const ray = new THREE.Raycaster();
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObject(surfaceMesh);
  const tip = document.getElementById("embedding-3d-tooltip");
  if (hits.length > 0) {
    const vIdx = hits[0].face.a;
    const ci = vIdx;
    const g = cellGenre[ci];
    const cnt = Math.round(cellCount[ci]);
    const genreName = g >= 0 ? state.genres[g].name : "Untagged";
    let html = `<div class="tip-label">${escapeHtml(genreName)}</div>`;
    if (cnt > 0) {
      html += `<span style="color:var(--muted)">${cnt.toLocaleString()} books in this region</span>`;
      if (window.__showBookNames) {
        const s = cellSamples[ci];
        if (s && s.length) {
          html += "<br>" + s.slice(0, 3).map(p =>
            `<b>${escapeHtml(p.title || "(untitled)")}</b><br>` +
            `<span style="color:var(--muted)">${escapeHtml(p.author || "Unknown")}</span>`
          ).join("<br>");
        }
      }
    } else {
      html += `<span style="color:var(--muted)">sparse region</span>`;
    }
    tip.innerHTML = html;
    tip.classList.remove("hidden");
    tip.style.left = (e.clientX + 14) + "px";
    tip.style.top = (e.clientY - 10) + "px";
  } else {
    tip.classList.add("hidden");
  }
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}

function animate() {
  if (!animating) return;
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

/* ---- public API for the Dots/Surface toggle ---- */
function show() {
  if (!initialized || !ready) return;
  if (animating) return;          // already looping
  animating = true;
  animate();
}
function hide() { animating = false; }

function renderLegend(data) {
  const stats = document.getElementById("embedding-3d-stats");
  if (stats && !stats.dataset.surface) {
    stats.dataset.surface = "1";
    const base = stats.innerHTML;
    stats.innerHTML = base + ` · <b>surface view</b>: height = book density, color = dominant genre`;
  }
  const legend = document.getElementById("embedding-3d-legend");
  if (legend && !legend.dataset.surface) {
    legend.dataset.surface = "1";
  }
}

async function bootstrap() {
  const wrap = document.getElementById("embedding-3d-wrap");
  const loading = document.createElement("div");
  loading.className = "embedding-3d-loading";
  loading.id = "embedding-3d-surface-loading";
  loading.textContent = "Loading surface…";
  wrap.appendChild(loading);
  try {
    state = await loadData();
  } catch (e) {
    loading.textContent = "Failed to load embedding data: " + e.message;
    loading.style.color = "var(--bad)";
    return;
  }
  init();
  buildSurface(state);
  renderLegend(state);
  loading.remove();
  initialized = true;
  ready = true;
  if (window.__reco3d) window.__reco3d.showSurfaceReady();
}

window.__surface = { ready: false, show, hide, setMode: (m) => { mode = m; applyMode(); } };

const observer = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) {
      observer.disconnect();
      bootstrap().then(() => {
        window.__surface.ready = true;
        const v = window.__embeddingView;
        if (v === "hills" || v === "flat") window.__surface.show();
      });
      break;
    }
  }
}, { rootMargin: "200px 0px" });
observer.observe(document.getElementById("embedding-3d-wrap"));
