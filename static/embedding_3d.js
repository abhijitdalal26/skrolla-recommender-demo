/* 3D embedding visualizer - real 3D point cloud.
   Lazy-loaded on scroll into view. Renders every book as a single dot at its
   actual 3D UMAP position (x, y, z), colored by dominant canonical genre. This
   shows the books' true vector-space geometry (how far apart things really are
   in 3D) instead of a smoothed sheet. Drag to rotate, scroll to zoom,
   right-drag to pan. Hover any dot to see its book and genre. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DATA_URL = "static/embedding_3d.json";
const BG_U = [0.043, 0.055, 0.078]; // matches --bg (#0b0e14)
let state = null;
let renderer = null;
let scene = null;
let camera = null;
let controls = null;
let cloud = null;                 // THREE.Points (one dot per book)
let animating = false;
let wantsVisible = true;
let initialized = false;
let resizeObserver = null;
let pointIndex = null;            // Points.index -> original point (for tooltips)

// Global toggle for showing book names in tooltips
window.__showBookNames = window.__showBookNames ?? true;

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

function init() {
  const canvas = document.getElementById("embedding-3d");
  const wrap = document.getElementById("embedding-3d-wrap");
  const W = wrap.clientWidth, H = 560;
  canvas.width = W; canvas.height = H;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(W, H, false);
  renderer.setClearColor(0x0b0e14, 1);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(55, W / H, 0.01, 100);
  camera.position.set(1.4, 0.9, 1.7);
  camera.lookAt(0, 0, 0);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.rotateSpeed = 0.55;
  controls.zoomSpeed = 0.7;
  controls.panSpeed = 0.6;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.4;

  renderer.domElement.addEventListener("pointermove", onPointerMove, false);
  renderer.domElement.addEventListener("pointerdown", () => { controls.autoRotate = false; });
  renderer.domElement.addEventListener("wheel", () => { controls.autoRotate = false; }, { passive: true });

  resizeObserver = new ResizeObserver(() => {
    const w = wrap.clientWidth;
    if (w === 0) return;
    camera.aspect = w / H;
    camera.updateProjectionMatrix();
    renderer.setSize(w, H, false);
  });
  resizeObserver.observe(wrap);
}

const POINT_SIZE = 0.009;

function buildCloud(data) {
  const n = data.points.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  pointIndex = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const p = data.points[i];
    positions[i * 3] = p.x;
    positions[i * 3 + 1] = p.y;
    positions[i * 3 + 2] = p.z;
    let c = [0.6, 0.62, 0.68];
    if (p.label >= 0 && p.label < data.genres.length) {
      const gc = hexToRgb(data.genres[p.label].color);
      c = gc;
    }
    colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
    pointIndex[i] = i;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  cloud = new THREE.Points(
    geom,
    new THREE.PointsMaterial({
      size: POINT_SIZE,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: false,
      depthWrite: true,
    })
  );
  scene.add(cloud);
}

function onPointerMove(e) {
  if (!cloud || !state) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1
  );
  const raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = POINT_SIZE * 2.2;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(cloud);
  const tip = document.getElementById("embedding-3d-tooltip");
  if (hits.length > 0) {
    const idx = hits[0].index ?? hits[0].pointIndex;
    let html;
    if (idx != null && pointIndex && state.points[idx]) {
      const p = state.points[idx];
      const g = p.label >= 0 ? state.genres[p.label] : null;
      const genreHtml = g ? escapeHtml(g.name) : "Untagged";
      html = `<div class="tip-label">${genreHtml}</div>`;
      if (window.__showBookNames) {
        html += `<b>${escapeHtml(p.title || "(untitled)")}</b><br>` +
          `<span style="color:var(--muted)">${escapeHtml(p.author || "Unknown")}</span>`;
      } else {
        html += `<span style="color:var(--muted)">hover with “Book names” on to see titles</span>`;
      }
    } else {
      html = `<div class="tip-label">Book</div><span style="color:var(--muted)">nearby</span>`;
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

function renderLegend(data) {
  const total = data.points.length;
  const stats = document.getElementById("embedding-3d-stats");
  const labeled = total - (data.genres.find(g => g.name === "Untagged")?.count || 0);
  stats.innerHTML =
    `<b>${total.toLocaleString()}</b> books · ` +
    `<b>${labeled.toLocaleString()}</b> genre-tagged (first-match-wins) · ` +
    `<b>${data.genres.length - 1}</b> canonical genres + untagged`;
  const legend = document.getElementById("embedding-3d-legend");
  legend.innerHTML = data.genres.map(g =>
    `<span class="dot" style="background:${g.color}"></span>${escapeHtml(g.name)} <span class="dim">${g.count.toLocaleString()}</span>`
  ).join("");
}

async function bootstrap() {
  const loading = document.getElementById("embedding-3d-loading");
  loading.textContent = "Loading 67,736 books…";
  try {
    state = await loadData();
  } catch (e) {
    loading.textContent = "Failed to load embedding data: " + e.message;
    loading.style.color = "var(--bad)";
    return;
  }
  loading.remove();
  init();
  buildCloud(state);
  renderLegend(state);
  initialized = true;
  if ((window.__embeddingView || "continuous") === "continuous" && wantsVisible) {
    animating = true;
    animate();
  }
}

// Lazy-load on scroll into view
const observer = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) {
      observer.disconnect();
      bootstrap();
      break;
    }
  }
}, { rootMargin: "200px 0px" });
observer.observe(document.getElementById("embedding-3d-wrap"));

/* ---- public API for the view switcher ---- */
function show() {
  wantsVisible = true;
  if (!initialized || animating) return;
  animating = true;
  animate();
}
function hide() { wantsVisible = false; animating = false; }
window.__continuous = { show, hide };
