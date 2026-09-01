/* Skrolla Recommender — static demo with browser-side taste.
 *
 * Unlike the offline Flask sim, this is a fully static site:
 *   - Covers stream from Cloudflare R2.
 *   - Read-only Supabase RPCs (match_books_nearest, get_genre_anchors,
 *     search_books, get_books_by_isbns, random_popular_seed) do the ANN work
 *     against the pgvector/HNSW catalog.
 *   - The taste profile (clusters, weights, signals) lives ENTIRELY in browser
 *     memory for this session. It is never stored or sent to the server.
 *     Reload resets it.
 *
 * The distance-ring serving model and dynamic clustering mirror the logic the
 * live app uses (see CHANGELOG 2026-08-26).
 */
"use strict";

// ---- Supabase config (anon key is public by design; RLS + read-only RPCs guard it) ----
const SUPABASE_URL = "https://hkdtpwbmgmnxhykhgqvb.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhrZHRwd2JtZ21ueGh5a2hncXZiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODgyMDUzMzcsImV4cCI6MjEwMzc4MTMzN30.fp3NGMdYZpA1o159ew6xHFdqdPpUMoRBbUD3fH70rCw";
const COVER_BASE = "https://pub-77f77d9160fd473c8b4e4e7ca1aa2e18.r2.dev";

// ---- taste model constants (mirror reco_engine / production) ----
const MERGE_DIST = 0.45;      // cosine distance to merge into a cluster
const MAX_CLUSTERS = 5;
const DECAY = 0.95;           // per-interaction weight decay
const WEIGHTS = { like: 1, save: 1, chat: 1, tap: 0.5, skip: -0.3, not_for_me: -0.5 };
const QUALITY_TAG_STRICT = "Strict";

let sb = null;
let state = null;
let feed = [];
let mode = "fyp";
let selectedGenres = new Set();
let lovedBooks = new Map();
let lastSeed = 0;
let feedIndex = 0;
let genres = [];
let genreAnchors = {};        // genre -> {mean, sub1..sub5}
let histStack = [];           // undo history of {snapshot, isbn, signal}
let fetching = false;

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ vector math
function toVec(s) {
  if (Array.isArray(s)) return s;
  if (typeof s === "string") { try { return JSON.parse(s); } catch { return null; } }
  return null;
}
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function norm(a) { return Math.sqrt(dot(a, a)) || 1; }
function cosineDist(a, b) { return 1 - dot(a, b) / (norm(a) * norm(b)); }
function cosineSim(a, b) { return 1 - cosineDist(a, b); }

// ------------------------------------------------------------------ state
function freshState() {
  return {
    has_taste: false,
    coverage: 0,
    total_swipes: 0,
    positive_signals: 0,
    clusters: [],
    seen: new Set(),
    blacklist: new Set(),
    tick: 0,
    signals: [],
    allocation: { inner_n: 0, middle_n: 0, outer_n: 0, mode: "no-taste", clusters_selected: 0 },
    slices: {},
  };
}

function labelFor(vec) {
  // nearest canonical genre anchor (by mean_vec) gives the cluster label
  let best = "taste", bestD = Infinity;
  for (const [g, a] of Object.entries(genreAnchors)) {
    const mv = toVec(a.mean_vec);
    if (!mv) continue;
    const d = cosineDist(vec, mv);
    if (d < bestD) { bestD = d; best = g; }
  }
  return best;
}

function decayAll() {
  for (const c of state.clusters) c.weight *= DECAY;
}

// Add a book vector as a signal. dir=+1 positive, -1 negative.
function addSignal(vec, dir) {
  if (!vec) return;
  decayAll();
  let nearest = null, bestD = Infinity;
  for (const c of state.clusters) {
    const d = cosineDist(vec, c.centroid);
    if (d < bestD) { bestD = d; nearest = c; }
  }
  if (nearest && bestD <= MERGE_DIST) {
    nearest.size += 1;
    nearest.weight += WEIGHTS[(dir > 0 ? "like" : "skip")];
    const nw = nearest.size;
    nearest.centroid = nearest.centroid.map((v, i) => v + (vec[i] - v) / nw);
  } else if (state.clusters.length < MAX_CLUSTERS) {
    state.clusters.push({
      id: state.clusters.length,
      centroid: vec.slice(),
      weight: dir > 0 ? 1 : -0.5,
      size: 1,
      label: labelFor(vec),
      genre_dist: bestD,
    });
  } else {
    // cap reached: reinforce the nearest cluster instead
    nearest.weight += WEIGHTS[(dir > 0 ? "like" : "skip")];
    nearest.size += 1;
  }
  state.clusters.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
}

function buildTasteFromOnboarding(genresList, lovedVecs) {
  // Seed from the chosen genres' sub-centroids (union), like apply_genre_seeds.
  const seeds = [];
  const gDocs = [];
  for (const g of genresList) {
    const a = genreAnchors[g];
    if (!a) continue;
    for (let i = 1; i <= 5; i++) {
      const v = toVec(a["sub" + i + "_vec"]);
      if (v) { seeds.push(v); gDocs.push(g); }
    }
    const m = toVec(a.mean_vec);
    if (m) { seeds.push(m); gDocs.push(g); }
  }
  for (const v of lovedVecs) if (v) { seeds.push(v); gDocs.push("loved"); }

  state.clusters = kmeans(seeds, gDocs, Math.min(MAX_CLUSTERS, Math.max(1, seeds.length)));
  state.has_taste = state.clusters.length > 0;
  state.coverage = state.has_taste ? 1 : 0;
}

function kmeans(points, docs, k) {
  if (!points.length) return [];
  k = Math.max(1, Math.min(k, points.length));
  // farthest-point init
  const centers = [0];
  const idxs = new Set([0]);
  while (centers.length < k) {
    let best = -1, bestD = -1;
    for (let i = 0; i < points.length; i++) {
      if (idxs.has(i)) continue;
      let md = Infinity;
      centers.forEach((c) => { const d = cosineDist(points[i], points[c]); if (d < md) md = d; });
      if (md > bestD) { bestD = md; best = i; }
    }
    if (best < 0) break;
    centers.push(best); idxs.add(best);
  }
  const assign = new Array(points.length).fill(0);
  const ncent = new Array(centers.length).fill(0).map(() => new Array(points[0].length).fill(0));
  for (let iter = 0; iter < 20; iter++) {
    assign.fill(0);
    for (let i = 0; i < points.length; i++) {
      let bc = 0, bd = Infinity;
      for (let c = 0; c < centers.length; c++) {
        const d = cosineDist(points[i], ncent[c].some((x) => x !== 0) ? ncent[c] : points[centers[c]]);
        if (d < bd) { bd = d; bc = c; }
      }
      assign[i] = bc;
    }
    const sums = centers.map(() => new Array(points[0].length).fill(0));
    const counts = centers.map(() => 0);
    for (let i = 0; i < points.length; i++) {
      const c = assign[i];
      counts[c]++;
      for (let j = 0; j < points[i].length; j++) sums[c][j] += points[i][j];
    }
    for (let c = 0; c < centers.length; c++) {
      if (counts[c]) { ncent[c] = sums[c].map((v) => v / counts[c]); }
      else ncent[c] = points[centers[c]].slice();
    }
  }
  const clusters = [];
  for (let c = 0; c < centers.length; c++) {
    const members = [];
    for (let i = 0; i < assign.length; i++) if (assign[i] === c) members.push(i);
    const major = (docs[members[0]] || "taste");
    clusters.push({
      id: c,
      centroid: ncent[c],
      weight: members.length,
      size: members.length,
      label: major,
      genre_dist: 0,
    });
  }
  clusters.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  return clusters;
}

// ------------------------------------------------------------------ Supabase RPC
function rpc(fn, args) {
  return sb.rpc(fn, args).then(({ data, error }) => {
    if (error) throw new Error(`${fn}: ${error.message}`);
    return data;
  });
}

// ------------------------------------------------------------------ feed building
function coverUrl(f) {
  return f.cover_url || (f.cover_file ? COVER_BASE + "/" + f.cover_file : "");
}

// strict quality gate (mirrors the live app)
function qualityPass(r) {
  if ((r.ratings_count || 0) >= 5000) return true;
  if (r.is_nyt_bestseller) return true;
  if (r.src === "manga" && (r.avg_rating || 0) >= 4.0) return true;
  if (r.src === "scraped") return true;
  return false;
}

function decodeRows(rows, slice) {
  return (rows || []).map((r) => ({
    isbn: r.isbn13,
    title: r.title,
    author: r.author,
    genres: r.genres || "",
    description: r.description || "",
    pub_year: r.pub_year,
    avg_rating: r.avg_rating != null ? +r.avg_rating.toFixed(2) : null,
    ratings_count: r.ratings_count != null ? +r.ratings_count : null,
    src: r.src,
    is_nyt_bestseller: !!r.is_nyt_bestseller,
    slice: slice,
    dist: r.distance != null ? +(+r.distance).toFixed(3) : null,
    cluster_label: r.cluster_label || null,
    cover_url: coverUrl(r),
    vec: toVec(r.combined_vec),
  }));
}

async function fetchNearest(queryVec, k, seenIsbns) {
  return rpc("match_books_nearest", {
    query_vec: queryVec,
    seen_isbns: seenIsbns,
    k: k,
    p_ef_search: 100,
  });
}

function topClusters(n) {
  return state.clusters.slice(0, n);
}

async function buildFeedInner(target, seenArr) {
  const sel = [];
  // top-2 by weight + 1 random from the rest
  const top = topClusters(2);
  sel.push(...top);
  const rest = state.clusters.slice(2);
  if (rest.length) { const r = rest[Math.floor(Math.random() * rest.length)]; sel.push(r); }
  if (sel.length === 0) return { rows: [], picked: 0, ann_returned: 0, clusters_used: 0 };

  const per = Math.max(8, Math.ceil((target * 3) / sel.length));
  const results = [];
  let ann = 0;
  for (const c of sel) {
    const rows = await fetchNearest(c.centroid, per, seenArr);
    ann += rows.length;
    rows.forEach((r) => { r.cluster_label = c.label; r.__club = c; });
    results.push(...rows);
  }
  const gated = results.filter((r) => qualityPass(r));
  const uniq = dedupe(gated);
  return { rows: uniq, picked: uniq.length, ann_returned: ann, clusters_used: sel.length };
}

async function buildFeedMiddle(target, seenArr) {
  if (!state.clusters.length) return { rows: [], picked: 0, sampled: 0, in_band: 0 };
  const top = state.clusters[0];
  const pool0 = await fetchNearest(top.centroid, 200, seenArr);
  const bandMin = 0.30, bandMax = 0.55;
  const inBand = pool0.filter((r) => {
    const d = r.distance != null ? +r.distance : cosineDist(r.combined_vec ? toVec(r.combined_vec) : top.centroid, top.centroid);
    return d >= bandMin && d <= bandMax;
  });
  const uniq = dedupe(inBand.filter((r) => qualityPass(r)));
  return { rows: uniq, picked: uniq.length, sampled: pool0.length, in_band: inBand.length };
}

async function buildFeedOuter(target, seenArr) {
  const rows = await rpc("random_popular_seed", { k: Math.max(target * 3, 20), seen_isbns: seenArr });
  const uniq = dedupe(rows.filter((r) => qualityPass(r)));
  return { rows: uniq, picked: uniq.length, sampled: rows.length };
}

function dedupe(rows) {
  const seen = new Set();
  const out = [];
  for (const r of rows) {
    const key = r.isbn13 || r.isbn;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(r);
  }
  return out;
}

async function buildFeed() {
  const total = mode === "fyp" ? 18 : 12;
  const seenSet = new Set(state.seen);
  feed.forEach((f) => seenSet.add(f.isbn));
  const seenArr = [...seenSet];

  let innerTarget = 0, middleTarget = 0, outerTarget = total;
  if (state.has_taste && state.clusters.length) {
    innerTarget = Math.round(total * 0.5);
    middleTarget = Math.round(total * 0.3);
    outerTarget = total - innerTarget - middleTarget;
  } else {
    middleTarget = Math.round(total * 0.5);
    outerTarget = total - middleTarget;
  }

  let inner = { rows: [], picked: 0, ann_returned: 0, clusters_used: 0 };
  let middle = { rows: [], picked: 0, sampled: 0, in_band: 0 };
  let outer = { rows: [], picked: 0, sampled: 0 };

  if (state.has_taste && innerTarget > 0) {
    inner = await buildFeedInner(innerTarget, seenArr);
  }
  if (middleTarget > 0) {
    middle = await buildFeedMiddle(middleTarget, seenArr);
  }
  if (outerTarget > 0) {
    outer = await buildFeedOuter(outerTarget, seenArr);
  }

  const chosen = [];
  const pick = (pool, n, slice) => {
    let i = 0;
    while (chosen.length < total && i < pool.length && chosen.filter((c) => c.slice === slice).length < n) {
      const r = pool[i];
      if (!state.seen.has(r.isbn13 || r.isbn) && !chosen.some((c) => (c.isbn13 || c.isbn) === (r.isbn13 || r.isbn))) {
        chosen.push(r);
      }
      i++;
    }
  };
  pick(inner.rows, innerTarget, "inner");
  pick(middle.rows, middleTarget, "middle");
  pick(outer.rows, outerTarget, "outer");
  // top up any shortfall from outer pool
  if (chosen.length < total) {
    const missing = total - chosen.length;
    pick(outer.rows, missing + outerTarget, "outer");
  }

  const stats = {
    allocation: {
      inner_n: chosen.filter((c) => c.slice === "inner").length,
      middle_n: chosen.filter((c) => c.slice === "middle").length,
      outer_n: chosen.filter((c) => c.slice === "outer").length,
      mode: state.has_taste ? "distance-ring" : "no-taste",
      clusters_selected: state.clusters.length,
    },
    slices: {
      inner: { ann_returned: inner.ann_returned, clusters_used: inner.clusters_used, gate_pass: inner.rows.length, picked: chosen.filter((c) => c.slice === "inner").length },
      middle: { sampled: middle.sampled, in_band: middle.in_band, gate_pass: middle.rows.length, picked: chosen.filter((c) => c.slice === "middle").length },
      outer: { sampled: outer.sampled, gate_pass: outer.rows.length, picked: chosen.filter((c) => c.slice === "outer").length },
    },
    returned: chosen.length,
  };
  state.allocation = stats.allocation;
  state.slices = stats.slices;
  return { feed: decodeRows(chosen, ""), stats };
}

// ------------------------------------------------------------------ UI wiring (mostly identical to the offline demo)
async function init() {
  sb = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  state = freshState();
  state.seen = new Set();
  state.blacklist = new Set();

  // load genre anchors for labels + onboarding seeds
  const anchors = await rpc("get_genre_anchors", {});
  genreAnchors = {};
  (anchors || []).forEach((a) => { genreAnchors[a.genre] = a; });
  genres = (anchors || []).map((a) => a.genre).filter(Boolean).sort();

  const grid = $("genre-grid");
  grid.innerHTML = genres.map((g) => `<button class="genre-chip" data-g="${escapeHtml(g)}">${escapeHtml(g)}</button>`).join("");
  grid.querySelectorAll(".genre-chip").forEach((el) => {
    el.addEventListener("click", () => {
      const g = el.dataset.g;
      if (selectedGenres.has(g)) { selectedGenres.delete(g); el.classList.remove("sel"); }
      else { selectedGenres.add(g); el.classList.add("sel"); }
      $("apply-onboarding").disabled = selectedGenres.size === 0;
    });
  });
  $("reset-btn").addEventListener("click", reset);
  $("apply-onboarding").addEventListener("click", applyOnboarding);
  $("more-btn").addEventListener("click", loadMore);
  $("prev-btn").addEventListener("click", () => moveFeed(-1));
  $("next-btn").addEventListener("click", () => moveFeed(1));
  $("tab-fyp").addEventListener("click", () => switchMode("fyp"));
  $("tab-disc").addEventListener("click", () => switchMode("disc"));
  $("tab-picked").addEventListener("click", () => switchMode("picked"));
  $("tab-search").addEventListener("click", () => switchMode("search"));
  $("search-input").addEventListener("input", debounce(onSearch, 250));
  $("right-search").addEventListener("input", debounce(onRightSearch, 250));
  $("discover-close").addEventListener("click", closeDiscover);
  $("discover-modal").querySelector(".modal-backdrop").addEventListener("click", closeDiscover);
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-box")) $("search-results").innerHTML = "";
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDiscover();
    if (mode !== "fyp" || !feed.length) return;
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (e.key === "ArrowDown" || e.key === "PageDown") { e.preventDefault(); moveFeed(1); }
    else if (e.key === "ArrowUp" || e.key === "PageUp") { e.preventDefault(); moveFeed(-1); }
  });
  renderState(); renderClusters(); renderSignals();
  await fetchFeed();
}

async function refresh() {
  renderState(); renderClusters(); renderSignals();
}

async function fetchFeed(seed = lastSeed) {
  if (fetching) return;
  fetching = true;
  try {
    const data = await buildFeed();
    // merge slice info from buildFeed's decoded rows
    feed = data.feed.map((f) => ({ ...f }));
    renderFeed();
    renderPipeline(data.stats, mode);
    renderSignals();
    // top up to target like the offline sim's loop
    let tries = 0;
    const target = mode === "fyp" ? 18 : 12;
    while (tries < 2 && data.feed.length < target && mode !== "picked") {
      lastSeed += 1;
      const more = await buildFeed();
      const have = new Set(feed.map((f) => f.isbn));
      const fresh = more.feed.filter((f) => !have.has(f.isbn));
      if (!fresh.length) break;
      feed = feed.concat(fresh);
      renderFeed();
      renderPipeline(more.stats, mode);
      tries += 1;
    }
  } finally {
    fetching = false;
  }
}

async function fetchPicked() {
  if (!state.has_taste || !state.clusters.length) {
    feed = [];
    renderFeed();
    renderPipeline({ note: "no-taste", allocation: {}, slices: {}, returned: 0 }, "picked");
    return;
  }
  const top = topClusters(3);
  const seen = [...state.seen];
  let rows = [];
  for (const c of top) {
    const r = await fetchNearest(c.centroid, 20, seen);
    rows.push(...r);
  }
  const uniq = dedupe(rows.filter((r) => qualityPass(r)));
  feed = decodeRows(uniq.slice(0, 12), "picked");
  renderFeed();
  renderPipeline({ allocation: {}, slices: {}, returned: feed.length }, "picked");
}

async function loadMore() {
  lastSeed += 1;
  const more = await buildFeed();
  const have = new Set(feed.map((f) => f.isbn));
  const fresh = more.feed.filter((f) => !have.has(f.isbn));
  feed = feed.concat(fresh);
  renderState(); renderFeed(); renderPipeline(more.stats, mode);
}

function switchMode(m) {
  mode = m;
  ["fyp", "disc", "picked", "search"].forEach((x) => $(`tab-${x}`).classList.toggle("active", x === m));
  const isFyp = m === "fyp";
  const isSearch = m === "search";
  $("feed-viewer").classList.toggle("hidden", !isFyp);
  $("feed").classList.toggle("hidden", isFyp || m === "picked" || isSearch);
  $("picked").classList.toggle("hidden", m !== "picked");
  $("search-out").classList.toggle("hidden", !isSearch);
  $("search-bar").classList.toggle("hidden", !isSearch);
  $("more-btn").classList.toggle("hidden", isSearch);
  if (isSearch) { $("right-search").focus(); return; }
  if (m === "picked") { fetchPicked(); return; }
  fetchFeed();
}

async function applyOnboarding() {
  const genresArr = [...selectedGenres];
  let lovedVecs = [];
  if (lovedBooks.size) {
    // fetch vectors for the loved books via get_books_by_isbns
    try {
      const isbns = [...lovedBooks.keys()];
      const rows = await rpc("get_books_by_isbns", { isbns });
      lovedVecs = (rows || []).map((r) => toVec(r.combined_vec)).filter(Boolean);
    } catch (e) { console.warn("loved vecs", e); }
  }
  buildTasteFromOnboarding(genresArr, lovedVecs);
  state.total_swipes += lovedBooks.size;
  state.positive_signals += lovedBooks.size;
  lovedBooks.clear();
  $("apply-onboarding").disabled = true;
  updateGenreChips();
  renderChips();
  renderState(); renderClusters(); renderSignals();
  await fetchFeed();
}

async function sendSignal(isbn, signal, btn) {
  const book = feed.find((f) => f.isbn === isbn);
  // snapshot for undo
  histStack.push({
    snapshot: JSON.parse(JSON.stringify({
      has_taste: state.has_taste, total_swipes: state.total_swipes,
      positive_signals: state.positive_signals, tick: state.tick, signals: state.signals,
      clusters: state.clusters.map((c) => ({ ...c, centroid: c.centroid.slice() })),
    })),
    isbn, signal,
  });
  state.total_swipes += 1;
  state.tick += 1;
  const w = WEIGHTS[signal] || 0;
  if (w > 0) state.positive_signals += 1;
  state.seen.add(isbn);
  state.signals.push({ tick: state.tick, signal, isbn });
  if (signal === "not_for_me") state.blacklist.add(isbn);
  if (book && book.vec) addSignal(book.vec, w >= 0 ? 1 : -1);
  if (btn) {
    btn.classList.add("done");
    btn.title = "Click to undo";
    btn.onclick = (e) => { e.stopPropagation(); undoSignal(isbn, btn); };
  }
  renderState(); renderClusters(); renderSignals();
}

async function undoSignal(isbn, btn) {
  const evt = histStack.pop();
  if (evt) {
    const s = evt.snapshot;
    state.has_taste = s.has_taste;
    state.total_swipes = s.total_swipes;
    state.positive_signals = s.positive_signals;
    state.tick = s.tick;
    state.signals = s.signals;
    state.clusters = s.clusters;
  }
  if (btn) {
    btn.classList.remove("done");
    btn.title = "";
    const sig = btn.dataset.sig;
    btn.onclick = (e) => { e.stopPropagation(); sendSignal(isbn, sig, btn); };
  }
  renderState(); renderClusters(); renderSignals();
  await fetchFeed(lastSeed);
}

async function reset() {
  state = freshState();
  state.seen = new Set();
  state.blacklist = new Set();
  feed = []; selectedGenres.clear(); lovedBooks.clear(); histStack = []; lastSeed = 0;
  $("onboarding-note").textContent = "";
  $("genre-grid").querySelectorAll(".genre-chip").forEach((el) => el.classList.remove("sel"));
  updateGenreChips();
  renderChips(); renderFeed(); renderPipeline(null, mode);
  closeDiscover();
  renderState(); renderClusters(); renderSignals();
  await fetchFeed();
}

async function onSearch() {
  const q = $("search-input").value.trim();
  const box = $("search-results");
  if (q.length < 2) { box.innerHTML = ""; return; }
  let rows = [];
  try { rows = await rpc("search_books", { q, p_limit: 10 }); } catch (e) { box.innerHTML = ""; return; }
  box.innerHTML = rows.map((r) => {
    const cu = coverUrl(r);
    return `<div class="sr-item" data-isbn="${escapeHtml(r.isbn13)}">
       ${cu ? `<img src="${cu}" alt="" onerror="this.style.display='none'">` : ""}
       <div><div class="t">${escapeHtml(r.title)}</div>
       <div class="a">${escapeHtml(r.author || "")} · ${escapeHtml(r.src || "")}</div></div>
     </div>`;
  }).join("") || `<div class="sr-item"><div class="t">No matches</div></div>`;
  box.querySelectorAll(".sr-item[data-isbn]").forEach((el) => {
    el.addEventListener("click", () => {
      const isbn = el.dataset.isbn;
      if (!lovedBooks.has(isbn) && lovedBooks.size < 5) {
        lovedBooks.set(isbn, { title: el.querySelector(".t").textContent, author: el.querySelector(".a").textContent });
      }
      box.innerHTML = ""; $("search-input").value = "";
      renderChips();
    });
  });
}

async function onRightSearch() {
  const q = $("right-search").value.trim();
  if (q.length < 2) { $("search-out").innerHTML = ""; $("search-count").textContent = ""; return; }
  let rows = [];
  try { rows = await rpc("search_books", { q, p_limit: 15 }); } catch (e) { rows = []; }
  feed = rows.map((r) => ({ ...decodeRows([r], "search")[0] }));
  $("search-count").textContent = `${rows.length} result(s)`;
  renderFeed();
}

function renderChips() {
  $("loved-chips").innerHTML = [...lovedBooks.entries()].map(([isbn, b]) =>
    `<span class="chip" data-isbn="${escapeHtml(isbn)}">${escapeHtml(b.title)}</span>`).join("");
  $("loved-chips").querySelectorAll(".chip").forEach((el) => {
    el.addEventListener("click", () => { lovedBooks.delete(el.dataset.isbn); renderChips(); });
  });
}

function updateGenreChips() {
  const grid = $("genre-grid");
  grid.querySelectorAll(".genre-chip").forEach((el) => { el.disabled = false; el.style.opacity = ""; });
}

function renderState() {
  if (!state) return;
  const b = [];
  b.push(`<span class="badge ${state.has_taste ? "warm" : ""}">taste: ${state.has_taste ? "built" : "empty"}</span>`);
  b.push(`<span class="badge warm">quality filter: ${QUALITY_TAG_STRICT}</span>`);
  $("state-badges").innerHTML = b.join("");
  renderSession();
}

function renderSession() {
  if (!state) return;
  const rows = [
    ["Coverage", `${(state.coverage * 100).toFixed(1)}%`],
    ["Signals", `${state.total_swipes}`],
    ["Positive", `${state.positive_signals}`],
    ["Clusters", `${state.clusters.length} / 5`],
  ];
  $("session-list").innerHTML = rows.map(([k, v]) =>
    `<div class="session-row"><span class="k">${k}</span><span class="v">${v}</span></div>`
  ).join("") +
  `<div class="session-tip">Coverage — how broad your taste is. Signals — books you acted on. Positive — likes, saves &amp; interested. Clusters — active taste clusters (max 5). Your taste is stored only in this browser tab for this session.</div>`;
}

function renderClusters() {
  const el = $("cluster-list");
  $("cluster-empty").style.display = state.clusters.length ? "none" : "";
  el.innerHTML = state.clusters.map((c) => {
    const neg = c.weight < 0;
    const w = Math.min(Math.abs(c.weight) * 50, 100);
    return `<div class="cluster">
      <div class="row"><span><b class="${neg ? "neg" : ""}">${escapeHtml((c.label || "?").replace(/[\s/|+-]+$/, ""))}</b> · ${c.size} sig</span>
        <span>w=${c.weight.toFixed(2)} · gd=${(c.genre_dist || 0).toFixed(2)}</span></div>
      <div class="bar"><div class="${neg ? "neg" : ""}" style="width:${w}%"></div></div>
    </div>`;
  }).join("");
}

function renderPipeline(stats, m) {
  const el = $("pipeline");
  if (!state) { el.innerHTML = ""; $("pipeline-empty").style.display = ""; return; }
  if (!stats || (stats.note === "no-taste" && m === "picked") || !state.clusters.length) {
    el.innerHTML = `<div class="hint">No taste profile yet. Pick genres in Onboarding to get started.</div>`;
    $("pipeline-empty").style.display = "none";
    return;
  }
  if (!stats) { el.innerHTML = ""; $("pipeline-empty").style.display = ""; return; }
  $("pipeline-empty").style.display = "none";
  const a = stats.allocation || {};
  let allocBar = "";
  if (m === "fyp") {
    const total = (a.inner_n || 0) + (a.middle_n || 0) + (a.outer_n || 0);
    allocBar = `<div class="pipe-alloc">
      <div class="inner" style="width:${(a.inner_n || 0) / total * 100 || 0}%"></div>
      <div class="middle" style="width:${(a.middle_n || 0) / total * 100 || 0}%"></div>
      <div class="outer" style="width:${(a.outer_n || 0) / total * 100 || 0}%"></div>
    </div>
    <div class="hint" style="margin-top:0">${a.mode || "?"} · cluster-count: ${a.clusters_selected ?? "?"}</div>`;
  }
  const slices = renderSliceStats(stats.slices, m);
  const note = m === "picked" ? `Picked: ANN against your taste centroids.`
                              : m === "disc" ? `Discovery: taste-close + far + curated random.`
                              : "";
  el.innerHTML = `${allocBar}${slices}
    <div class="hint">${note} returned: <b>${stats.returned ?? "?"}</b> books.</div>`;
}

function renderSliceStats(slices, m) {
  let out = "";
  const spec = m === "fyp"
    ? [["inner", "Inner · closest ANN (50%)"], ["middle", "Middle · 0.30-0.55 band (30%)"], ["outer", "Outer · random quality (20%)"]]
    : [["inner", "Taste close"], ["middle", "Explore band"], ["outer", "Random quality"]];
  for (const [key, label] of spec) {
    const s = (slices || {})[key] || {};
    if (key === "inner")
      out += sliceHtml(label, `clusters: <b>${s.clusters_used ?? 0}</b> · ann: <b>${s.ann_returned ?? 0}</b>`, `gate pass: <b>${s.gate_pass ?? 0}</b> · picked: <b class="hot">${s.picked ?? 0}</b>`);
    else if (key === "middle")
      out += sliceHtml(label, `sampled: <b>${s.sampled ?? 0}</b> · in band: <b>${s.in_band ?? 0}</b>`, `gate pass: <b>${s.gate_pass ?? 0}</b> · picked: <b class="hot">${s.picked ?? 0}</b>`);
    else
      out += sliceHtml(label, `sampled: <b>${s.sampled ?? 0}</b>`, `gate pass: <b>${s.gate_pass ?? 0}</b> · picked: <b class="hot">${s.picked ?? 0}</b>`);
  }
  return out;
}

function sliceHtml(label, left, right) {
  return `<div class="slice"><h3>${label}</h3><div class="kv"><span>${left}</span><span>${right}</span></div></div>`;
}

function renderSignals() {
  if (!state) return;
  const list = $("signal-list");
  const rows = state.signals.slice().reverse().map((s) => {
    const book = shortTitle(s.isbn);
    return `<div class="sig"><span class="t">#${s.tick}</span><span class="s ${s.signal}">${s.signal}</span>
      <span class="ttl">${escapeHtml(book)}</span></div>`;
  });
  list.innerHTML = rows.join("") || `<div class="hint">No signals yet.</div>`;
}

function renderFeed() {
  const gridEl = $("feed");
  const pickedEl = $("picked");
  const searchEl = $("search-out");
  const isFyp = mode === "fyp";
  if (!feed.length) {
    const msg = `<div class="empty">Select genres on the left to start exploring recommendations.</div>`;
    if (isFyp) { $("viewer").innerHTML = msg; $("viewer-counter").textContent = ""; }
    else if (mode === "picked") pickedEl.innerHTML = msg;
    else if (mode === "search") searchEl.innerHTML = msg;
    else gridEl.innerHTML = msg;
    return;
  }
  if (isFyp) {
    feedIndex = Math.max(0, Math.min(feed.length - 1, feedIndex));
    const f = feed[feedIndex];
    $("viewer").innerHTML = cardHtml(f, true);
    $("viewer-counter").textContent = `${feedIndex + 1} / ${feed.length}`;
    attachActions($("viewer"));
  } else if (mode === "picked") {
    pickedEl.innerHTML = feed.map((f) => cardHtml(f)).join("");
    attachActions(pickedEl);
  } else if (mode === "search") {
    searchEl.innerHTML = feed.map((f) => cardHtml(f)).join("");
    attachActions(searchEl);
  } else {
    gridEl.innerHTML = feed.map((f) => cardHtml(f)).join("");
    attachActions(gridEl);
  }
}

function moveFeed(dir) {
  if (mode !== "fyp" || !feed.length) return;
  const ni = Math.max(0, Math.min(feed.length - 1, feedIndex + dir));
  if (ni === feedIndex) {
    if (dir > 0) loadMore();
    return;
  }
  const viewer = $("viewer");
  viewer.classList.remove("slide-up", "slide-down");
  void viewer.offsetWidth;
  viewer.classList.add(dir > 0 ? "slide-up" : "slide-down");
  feedIndex = ni;
  const f = feed[feedIndex];
  viewer.innerHTML = cardHtml(f, true);
  $("viewer-counter").textContent = `${feedIndex + 1} / ${feed.length}`;
  attachActions(viewer);
}

function displayTitle(f, fyp) {
  let t = f.title || "";
  if (fyp) t = t.replace(/\s*\([^)]*\)\s*$/g, "").replace(/\s+/, " ").trim();
  return t || f.title || "";
}

function cardHtml(f, fyp) {
  const reason = buildReason(f);
  const title = displayTitle(f, fyp);
  const cu = f.cover_url;
  const coverHtml = cu
    ? `<img src="${cu}" alt="" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'no-cover\\'>${escapeHtml(f.title || 'No Cover')}</div>'">`
    : `<div class="no-cover">${escapeHtml(f.title || "No Cover")}</div>`;
  return `<div class="fbook" data-isbn="${escapeHtml(f.isbn)}">
    <div class="cov">
      ${coverHtml}
      ${f.slice ? `<span class="slot-tag ${f.slice}">${escapeHtml(f.slice)}</span>` : ""}
    </div>
    <div class="body">
      <div class="title" title="${escapeHtml(f.title || "")}">${escapeHtml(title)}</div>
      <div class="author">${escapeHtml(f.author || "")}</div>
      <div class="reason">${reason}</div>
      <div class="acts">
        <button class="save" data-sig="save">Save</button>
        <button class="like" data-sig="like">Like</button>
        <button class="interested" data-sig="tap">Interested</button>
        <button class="nfm" data-sig="not_for_me">Not for me</button>
      </div>
    </div>
  </div>`;
}

function attachActions(root) {
  root.querySelectorAll("button[data-sig]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const card = btn.closest(".fbook");
      sendSignal(card.dataset.isbn, btn.dataset.sig, btn);
    });
  });
  root.querySelectorAll(".title").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      const card = el.closest(".fbook");
      if (card) openDiscover(card.dataset.isbn);
    });
  });
}

function buildReason(f) {
  if (f.slice === "search") {
    const parts = ["<b>search</b>"];
    if (f.src) parts.push(`src: ${escapeHtml(f.src)}`);
    return parts.join(" · ");
  }
  const parts = [];
  parts.push(`<b>${escapeHtml(f.slice || "")}</b>`);
  if (f.cluster_label) parts.push(`cluster: ${escapeHtml(f.cluster_label)}`);
  if (f.dist != null) parts.push(`dist=${f.dist.toFixed(2)}`);
  return parts.join(" · ");
}

function shortTitle(isbn) {
  const f = feed.find((x) => x.isbn === isbn);
  return f && f.title ? (f.title.length > 30 ? f.title.slice(0, 28) + "..." : f.title) : isbn;
}

// ------------------------------------------------------------------ discover modal
function openDiscover(isbn) {
  const book = feed.find((x) => x.isbn === isbn);
  if (!book) return;
  const modal = $("discover-modal");
  const inner = modal.querySelector(".discover-book");
  const coverHtml = book.cover_url
    ? `<img src="${book.cover_url}" alt="${escapeHtml(book.title || "")}">`
    : `<div class="no-cover">${escapeHtml(book.title || "No Cover")}</div>`;
  const genresArr = book.genres ? (typeof book.genres === "string" ? (() => { try { return JSON.parse(book.genres); } catch { return []; } })() : book.genres) : [];
  const tags = genresArr.map((g) => `<span class="tag">${escapeHtml(g)}</span>`).join("");
  inner.innerHTML = `
    <div class="cover">${coverHtml}</div>
    <div class="info">
      <h1>${escapeHtml(book.title || "")}</h1>
      <div class="meta">
        <b>${escapeHtml(book.author || "Unknown author")}</b>
        ${book.pub_year ? ` · ${book.pub_year}` : ""}
        ${book.avg_rating ? ` · ${book.avg_rating.toFixed(1)} avg rating` : ""}
        ${book.ratings_count ? ` · ${book.ratings_count.toLocaleString()} ratings` : ""}
      </div>
      ${tags ? `<div class="tags">${tags}</div>` : ""}
      ${book.description ? `<div class="desc">${escapeHtml(book.description)}</div>` : ""}
      <div class="actions">
        <button class="like" onclick="sendSignal('${isbn}','like',null)">Like</button>
        <button class="save" style="background:var(--accent)" onclick="sendSignal('${isbn}','save',null)">Save</button>
        <button class="interested" style="background:var(--warn);color:#ffffff" onclick="sendSignal('${isbn}','tap',null)">Interested</button>
        <button class="nfm" style="background:var(--panel-2);border:1px solid var(--border);color:var(--text)" onclick="sendSignal('${isbn}','not_for_me',null)">Not for me</button>
      </div>
    </div>`;
  modal.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeDiscover() {
  $("discover-modal").classList.add("hidden");
  document.body.style.overflow = "";
}

// ------------------------------------------------------------------ utils
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

init().catch((e) => console.error(e));
