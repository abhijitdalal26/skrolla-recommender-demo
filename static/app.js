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
 * Serving model mirrors the PRODUCTION unified recommender
 * (20260902000000_unified_mmr_recommendations.sql):
 *   - Scoring: _recommend_score w_taste/w_genre/w_quality/w_freshness
 *     per source (taste 0.45/0.25/0.20/0.10 · trending 0.25/0.15/0.45/0.15
 *     · exploration 0.20/0.10/0.15/0.55) + disliked_centroid penalty.
 *   - Diversity: _mmr_rerank (lambda coverage-driven 0.35-0.85).
 *   - Sources: taste + exploration when warm, else trending + exploration.
 *   - Taste dynamics: _apply_taste_signal (decay 0.95, threshold 0.45,
 *     weight-scaled merge, L2-normalize, LRU evict) + _apply_disliked_signal
 *     (decay 0.90, separate centroid).
 *   - Onboarding: union sub1..5 only, _kmeans_halfvec init-spread → ≤5.
 */
"use strict";

// ---- Supabase config (anon key is public by design; RLS + read-only RPCs guard it) ----
const SUPABASE_URL = "https://hkdtpwbmgmnxhykhgqvb.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhrZHRwd2JtZ21ueGh5a2hncXZiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODgyMDUzMzcsImV4cCI6MjEwMzc4MTMzN30.fp3NGMdYZpA1o159ew6xHFdqdPpUMoRBbUD3fH70rCw";
const COVER_BASE = "https://pub-77f77d9160fd473c8b4e4e7ca1aa2e18.r2.dev";

// ---- taste model constants (mirror production) ----
const MERGE_DIST = 0.45;      // cosine distance to merge into a cluster
const MAX_CLUSTERS = 5;
const TASTE_DECAY = 0.95;     // per taste-signal weight decay (_apply_taste_signal)
const DISLIKED_DECAY = 0.90;  // per disliked-signal decay (_apply_disliked_signal)
const DISLIKED_PENALTY_W = 0.35;
const QUALITY_TAG_STRICT = "Strict";
const TOTAL_CATALOG = 74730;  // approx, for coverage_frac

let sb = null;
let state = null;
let feed = [];
let mode = "fyp";
let selectedGenres = new Set();
let lovedBooks = new Map();
let lastSeed = 0;
let feedIndex = 0;
let genres = [];
let genreAnchors = {};        // genre -> {mean_vec, sub1..sub5_vec}
let histStack = [];           // undo history
let fetching = false;

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ vector math
function toVec(s) {
  if (!s) return null;
  if (Array.isArray(s)) return s;
  if (typeof s === "string") { try { const v = JSON.parse(s); return Array.isArray(v) ? v : null; } catch { return null; } }
  return null;
}
function l2Normalize(v) {
  let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s);
  if (n === 0) return v.slice();
  const out = new Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function norm(a) { return Math.sqrt(dot(a, a)) || 1; }
function cosineSim(a, b) {
  if (!a || !b) return 0;
  const na = norm(a), nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}
function cosineDist(a, b) { return 1 - cosineSim(a, b); }
function scaleVec(v, s) { const out = new Array(v.length); for (let i = 0; i < v.length; i++) out[i] = v[i] * s; return out; }
function addVec(a, b) { const out = new Array(a.length); for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i]; return out; }

// ------------------------------------------------------------------ state
function freshState() {
  return {
    has_taste: false,
    coverage: 0,
    total_swipes: 0,
    positive_signals: 0,
    clusters: [],            // {id, centroid: number[], weight, size, label, genre_dist, updated_at}
    disliked: null,          // {vec: number[], weight}
    shownCounts: {},         // isbn -> int (for freshness)
    seen: new Set(),
    blacklist: new Set(),
    tick: 0,
    nextId: 1,
    signals: [],
    allocation: { mode: "no-taste", lambda: 0, coverage: 0, clusters_selected: 0, sources: [] },
    slices: {},
  };
}

function labelFor(vec) {
  let best = "taste", bestD = Infinity;
  for (const [g, a] of Object.entries(genreAnchors)) {
    const mv = toVec(a.mean_vec);
    if (!mv) continue;
    const d = cosineDist(vec, mv);
    if (d < bestD) { bestD = d; best = g; }
  }
  return best;
}

// ---- _apply_taste_signal (prod: 20260814050925, decay 0.95, threshold 0.45, weight-scaled merge, LRU) ----
function applyTasteSignal(vec) {
  if (!vec) return;
  const w = 1;
  // decay existing weights (deployed does this regardless of outcome, before merge)
  for (const c of state.clusters) c.weight *= TASTE_DECAY;
  state.tick += 1;
  let nearest = null, bestD = Infinity;
  for (const c of state.clusters) {
    const d = cosineDist(vec, c.centroid);
    if (d < bestD) { bestD = d; nearest = c; }
  }
  if (nearest && bestD <= MERGE_DIST) {
    const oldW = nearest.weight; // already decayed
    const newW = oldW + w;
    // weight-scaled merge then L2 normalize (prod: _l2_normalize(centroid*old_w + vec*w))
    const merged = l2Normalize(addVec(scaleVec(nearest.centroid, oldW), scaleVec(vec, w)));
    nearest.centroid = merged;
    nearest.weight = newW;
    nearest.size = (nearest.size || 1) + 1;
    nearest.updated_at = state.tick;
    // label stays as originally assigned; could refresh but keep stable
  } else {
    if (state.clusters.length >= MAX_CLUSTERS) {
      // LRU evict: smallest updated_at (oldest), tie-break smallest weight
      let evictIdx = 0;
      for (let i = 1; i < state.clusters.length; i++) {
        const a = state.clusters[i], b = state.clusters[evictIdx];
        if (a.updated_at < b.updated_at || (a.updated_at === b.updated_at && a.weight < b.weight)) evictIdx = i;
      }
      state.clusters.splice(evictIdx, 1);
    }
    const nid = state.nextId++;
    state.clusters.push({
      id: nid,
      centroid: l2Normalize(vec.slice()),
      weight: w,
      size: 1,
      label: labelFor(vec),
      genre_dist: bestD === Infinity ? 0 : bestD,
      updated_at: state.tick,
    });
  }
  // prune near-zero
  state.clusters = state.clusters.filter((c) => c.weight > 1e-9);
  state.clusters.sort((a, b) => b.weight - a.weight);
  state.has_taste = state.clusters.length > 0;
  state.coverage = state.has_taste ? Math.min(1, state.seen.size / TOTAL_CATALOG) : 0;
}

// ---- _apply_disliked_signal (prod: 20260901041448, decay 0.90, separate centroid) ----
function applyDislikedSignal(vec) {
  if (!vec) return;
  const nvec = l2Normalize(vec.slice());
  if (!state.disliked) {
    state.disliked = { vec: nvec, weight: 1 };
  } else {
    // prod: v_centroid = scale(old,0.90)+scale(vec,1.0); weight similarly (centroid not normalized in DB)
    // For cosine scoring direction matters, so we normalize after combining.
    const scaledOld = scaleVec(state.disliked.vec, DISLIKED_DECAY);
    let nv = addVec(scaledOld, nvec);
    nv = l2Normalize(nv);
    state.disliked.vec = nv;
    state.disliked.weight = state.disliked.weight * DISLIKED_DECAY + 1;
  }
  state.tick += 1;
}

function kmeansHalfvec(vecs, k) {
  if (!vecs.length || k === 0) return [];
  k = Math.min(k, vecs.length);
  // L2 normalize all
  const normed = vecs.map((v) => l2Normalize(v));
  const n = normed.length;
  // Init: spread picks evenly across the pool (prod _kmeans_halfvec: (i*n)//k)
  let centroids = [];
  for (let i = 0; i < k; i++) centroids.push(normed[Math.floor((i * n) / k)].slice());
  for (let iter = 0; iter < 20; iter++) {
    const assign = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) {
        const d = cosineDist(normed[i], centroids[c]);
        if (d < bd) { bd = d; best = c; }
      }
      assign[i] = best;
    }
    const newCentroids = centroids.map(() => null);
    let changed = false;
    for (let c = 0; c < k; c++) {
      const members = [];
      for (let i = 0; i < n; i++) if (assign[i] === c) members.push(normed[i]);
      if (!members.length) { newCentroids[c] = centroids[c].slice(); continue; }
      let mean = new Array(normed[0].length).fill(0);
      for (const m of members) for (let d = 0; d < mean.length; d++) mean[d] += m[d];
      for (let d = 0; d < mean.length; d++) mean[d] /= members.length;
      mean = l2Normalize(mean);
      newCentroids[c] = mean;
      if (cosineDist(mean, centroids[c]) > 1e-6) changed = true;
    }
    centroids = newCentroids;
    if (!changed) break;
  }
  return centroids;
}

function buildTasteFromOnboarding(genresList, lovedVecs) {
  // Union sub1..5 only (no mean) like prod apply_genre_seeds; kmeans to ≤5.
  const seeds = [];
  for (const g of genresList) {
    const a = genreAnchors[g];
    if (!a) continue;
    for (let i = 1; i <= 5; i++) {
      const v = toVec(a["sub" + i + "_vec"]);
      if (v) seeds.push(l2Normalize(v));
    }
  }
  for (const v of lovedVecs) if (v) seeds.push(l2Normalize(v));
  if (!seeds.length) { state.clusters = []; state.has_taste = false; return; }
  const kFinal = Math.min(MAX_CLUSTERS, seeds.length);
  const cents = kmeansHalfvec(seeds, kFinal);
  state.clusters = [];
  state.nextId = 1;
  state.tick = 0;
  for (const c of cents) {
    state.clusters.push({
      id: state.nextId++,
      centroid: l2Normalize(c.slice()),
      weight: 1,
      size: Math.ceil(seeds.length / kFinal),
      label: labelFor(c),
      genre_dist: 0,
      updated_at: state.tick++,
    });
  }
  state.clusters.sort((a, b) => b.weight - a.weight);
  state.has_taste = state.clusters.length > 0;
  state.coverage = state.has_taste ? 1 : 0;
  state.disliked = null;
  state.shownCounts = {};
}

// ------------------------------------------------------------------ scoring + MMR (prod _recommend_score / _mmr_rerank)
function parseGenres(g) {
  if (!g) return [];
  if (Array.isArray(g)) return g;
  if (typeof g === "string") {
    const s = g.trim();
    if (s.startsWith("[")) { try { const a = JSON.parse(s); return Array.isArray(a) ? a : []; } catch { /* fall through */ } }
    // fallback comma split
    return s.split(",").map((x) => x.trim()).filter(Boolean);
  }
  return [];
}

function qualityScore(avg_rating, ratings_count, is_nyt) {
  let q = 0;
  if (avg_rating != null) q += (avg_rating / 5) * 0.4;
  else q += 0.15;
  if (ratings_count != null) {
    const lc = Math.log10(Math.max(ratings_count, 1));
    q += Math.min(lc / 5, 1) * 0.4; // 1k->0.24, 10k->0.32, 100k->0.4
  } else q += 0.1;
  if (is_nyt) q += 0.2;
  return Math.min(q, 1);
}
function freshnessScore(shownCount) {
  return 1 / (1 + Math.log(1 + shownCount));
}
function tasteSimilarity(bookVec, clusters) {
  if (!bookVec || !clusters.length) return 0;
  let best = -1;
  for (const c of clusters) {
    if (c.weight <= 0) continue;
    const s = cosineSim(bookVec, c.centroid);
    if (s > best) best = s;
  }
  return best < 0 ? 0 : best; // 0..1 (normalized vecs => cosine in -1..1, clamp neg to 0)
}
function dislikedPenalty(bookVec, disliked) {
  if (!bookVec || !disliked || !disliked.vec) return 0;
  const s = cosineSim(bookVec, disliked.vec);
  return Math.max(0, s) * DISLIKED_PENALTY_W; // 0..0.35
}
function genreMatchScore(bookGenresArr, pickedGenres) {
  if (!bookGenresArr.length || !pickedGenres.length) return 0;
  const pickedSet = new Set(pickedGenres.map((x) => x.toLowerCase()));
  for (const g of bookGenresArr) if (pickedSet.has(String(g).toLowerCase())) return 1;
  // also substring fallback: e.g. "Comics & Graphic Novels" vs "comic"
  const joined = bookGenresArr.join(" ").toLowerCase();
  for (const pg of pickedSet) if (joined.includes(pg)) return 1;
  return 0;
}
function getSourceWeights(source) {
  // mirror _recommend_score per-source weights
  if (source === "taste") return { w_taste: 0.45, w_genre: 0.25, w_quality: 0.20, w_fresh: 0.10 };
  if (source === "trending") return { w_taste: 0.25, w_genre: 0.15, w_quality: 0.45, w_fresh: 0.15 };
  // exploration
  return { w_taste: 0.20, w_genre: 0.10, w_quality: 0.15, w_fresh: 0.55 };
}
function recommendScore(book, bookVec, source) {
  const { w_taste, w_genre, w_quality, w_fresh } = getSourceWeights(source);
  const bookGenres = parseGenres(book.genres);
  const picked = [...selectedGenres];
  const shown = state.shownCounts[book.isbn13 || book.isbn] || 0;
  const q = qualityScore(book.avg_rating, book.ratings_count, !!book.is_nyt_bestseller);
  const f = freshnessScore(shown);
  const tRaw = tasteSimilarity(bookVec, state.clusters);
  const dPen = dislikedPenalty(bookVec, state.disliked);
  const t = Math.max(-1, tRaw - dPen); // taste component after penalty
  const gM = genreMatchScore(bookGenres, picked);
  let score = w_taste * t + w_genre * gM + w_quality * q + w_fresh * f;
  if (score < 0) score = 0;
  return { score, comp: { tRaw, dPen, t, gM, q, f }, bookGenres };
}
function mmrRerank(cands, k, lambda) {
  // cands: [{isbn, vec, score, source, row}]
  if (!cands.length || k <= 0) return [];
  // sort by score desc for deterministic tie-break
  cands = cands.slice().sort((a, b) => b.score - a.score);
  const picked = [];
  const remaining = cands.slice();
  while (picked.length < k && remaining.length) {
    let bestIdx = 0, bestMmr = -Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const cand = remaining[i];
      let maxSim = 0;
      if (picked.length && cand.vec) {
        for (const p of picked) {
          if (!p.vec || !cand.vec) continue;
          const s = cosineSim(cand.vec, p.vec);
          if (s > maxSim) maxSim = s;
        }
      }
      const mmr = lambda * cand.score - (1 - lambda) * maxSim;
      if (mmr > bestMmr) { bestMmr = mmr; bestIdx = i; }
    }
    picked.push(remaining[bestIdx]);
    remaining.splice(bestIdx, 1);
  }
  return picked;
}

// ------------------------------------------------------------------ Supabase RPC
function rpc(fn, args) {
  return sb.rpc(fn, args).then(({ data, error }) => {
    if (error) throw new Error(`${fn}: ${error.message}`);
    return data;
  });
}

// ------------------------------------------------------------------ feed building (UNIFIED)
function coverUrl(f) {
  return f.cover_url || (f.cover_file ? COVER_BASE + "/" + f.cover_file : "");
}

function decodeRows(rows, slice) {
  return (rows || []).map((r) => ({
    isbn: r.isbn13 || r.isbn,
    title: r.title,
    author: r.author,
    genres: r.genres || "",
    description: r.description || "",
    pub_year: r.pub_year,
    avg_rating: r.avg_rating != null ? +(+r.avg_rating).toFixed(2) : null,
    ratings_count: r.ratings_count != null ? +r.ratings_count : null,
    src: r.src,
    is_nyt_bestseller: !!r.is_nyt_bestseller,
    slice: slice || r.__source || "",
    source: r.__source || slice || "",
    dist: r.distance != null ? +(+r.distance).toFixed(3) : null,
    cluster_label: r.cluster_label || null,
    cover_url: coverUrl(r),
    vec: toVec(r.combined_vec),
    _raw: r,
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

  const hasTaste = state.has_taste && state.clusters.some((c) => c.weight > 0);
  const cov = state.seen.size / TOTAL_CATALOG;
  const EXPLORE_FLOOR = 0.25, EXPLORE_CEIL = 0.90, TARGET_COV = 0.01;
  let lambda = EXPLORE_FLOOR + (EXPLORE_CEIL - EXPLORE_FLOOR) * Math.max(0, 1 - cov / TARGET_COV);
  lambda = Math.max(0.35, Math.min(0.85, lambda));
  // warm users converge toward relevance; cold users keep diversity high
  if (!hasTaste) lambda = Math.max(lambda, 0.55);

  const sources = hasTaste ? ["taste", "exploration"] : ["trending", "exploration"];
  const poolCap = Math.max(total * 4, 60);

  let annReturned = 0, trendingReturned = 0, explorationReturned = 0;
  let candidatesRaw = [];

  if (sources.includes("taste")) {
    const top = state.clusters.filter((c) => c.weight > 0).sort((a, b) => b.weight - a.weight).slice(0, 3);
    for (const c of top) {
      const rows = await fetchNearest(c.centroid, poolCap, seenArr);
      annReturned += rows.length;
      rows.forEach((r) => { r.__source = "taste"; r.__cluster = c; });
      candidatesRaw.push(...rows);
    }
  }
  if (sources.includes("trending")) {
    const rows = await rpc("random_popular_seed", { k: poolCap, seen_isbns: seenArr });
    trendingReturned = rows.length;
    rows.forEach((r) => { r.__source = "trending"; });
    candidatesRaw.push(...rows);
  }
  if (sources.includes("exploration")) {
    const rows = await rpc("random_popular_seed", { k: poolCap, seen_isbns: seenArr });
    explorationReturned = rows.length;
    rows.forEach((r) => { r.__source = "exploration"; });
    candidatesRaw.push(...rows);
  }

  // dedupe before scoring (series dedup not needed client-side)
  const uniqMap = new Map();
  for (const r of candidatesRaw) {
    const key = r.isbn13 || r.isbn;
    if (!uniqMap.has(key)) uniqMap.set(key, r);
  }
  let uniq = [...uniqMap.values()];

  // blacklist filter + seen filter (server already filtered by seen, but double-check)
  uniq = uniq.filter((r) => {
    const k2 = r.isbn13 || r.isbn;
    return !state.blacklist.has(k2) && !state.seen.has(k2);
  });

  // score each candidate per its source weights
  const scored = [];
  for (const r of uniq) {
    const vec = toVec(r.combined_vec);
    const src = r.__source || (hasTaste ? "taste" : "trending");
    const { score, comp } = recommendScore(r, vec, src);
    scored.push({ isbn: r.isbn13 || r.isbn, vec, score, source: src, row: r, comp });
  }
  scored.sort((a, b) => b.score - a.score);

  // take top pool before MMR (keep more than total for diversity)
  const preMmr = scored.slice(0, Math.min(scored.length, total * 4));

  // MMR rerank to final total
  const picked = mmrRerank(preMmr, total, lambda);

  // update shown counts for freshness
  for (const p of picked) {
    state.shownCounts[p.isbn] = (state.shownCounts[p.isbn] || 0) + 1;
  }

  const decoded = picked.map((p) => {
    const r = p.row;
    const d = decodeRows([r], p.source)[0];
    d.score = +p.score.toFixed(3);
    d.mmrSource = p.source;
    // keep vec for potential future scoring
    d.vec = p.vec;
    // annotate comps for debug
    d._comp = p.comp;
    return d;
  });

  const stats = {
    allocation: {
      mode: hasTaste ? "unified-taste+explore" : "unified-trending+explore",
      sources,
      lambda: +lambda.toFixed(2),
      coverage: +cov.toFixed(4),
      clusters_selected: hasTaste ? Math.min(3, state.clusters.filter((c) => c.weight > 0).length) : 0,
      clusters_total: state.clusters.length,
    },
    slices: {
      taste: { ann_returned: annReturned, pool: scored.filter((s) => s.source === "taste").length, picked: picked.filter((p) => p.source === "taste").length },
      trending: { sampled: trendingReturned, pool: scored.filter((s) => s.source === "trending").length, picked: picked.filter((p) => p.source === "trending").length },
      exploration: { sampled: explorationReturned, pool: scored.filter((s) => s.source === "exploration").length, picked: picked.filter((p) => p.source === "exploration").length },
    },
    scoring: {
      scored_total: scored.length,
      pre_mmr: preMmr.length,
    },
    returned: decoded.length,
  };
  state.allocation = stats.allocation;
  state.slices = stats.slices;
  return { feed: decoded, stats };
}

// ------------------------------------------------------------------ UI wiring
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
    feed = data.feed.map((f) => ({ ...f }));
    feedIndex = 0;
    renderFeed();
    renderPipeline(data.stats, mode);
    renderSignals();
  } finally {
    fetching = false;
  }
}

async function fetchPicked() {
  // Picked tab: top ANN from taste clusters, scored + MMR (no exploration)
  if (!state.has_taste || !state.clusters.length) {
    feed = [];
    renderFeed();
    renderPipeline({ note: "no-taste", allocation: {}, slices: {}, returned: 0 }, "picked");
    return;
  }
  const seen = [...state.seen, ...feed.map((f) => f.isbn)];
  let rows = [];
  const top = state.clusters.filter((c) => c.weight > 0).sort((a, b) => b.weight - a.weight).slice(0, 3);
  for (const c of top) {
    const r = await fetchNearest(c.centroid, 24, seen);
    r.forEach((x) => { x.__source = "taste"; });
    rows.push(...r);
  }
  const uniq = dedupe(rows);
  // score + MMR
  const scored = uniq.map((r) => {
    const vec = toVec(r.combined_vec);
    const { score } = recommendScore(r, vec, "taste");
    return { isbn: r.isbn13 || r.isbn, vec, score, source: "taste", row: r };
  }).sort((a, b) => b.score - a.score);
  const picked = mmrRerank(scored, 12, 0.55);
  feed = picked.map((p) => {
    const d = decodeRows([p.row], "picked")[0];
    d.score = +p.score.toFixed(3);
    d.source = p.source;
    d.vec = p.vec;
    return d;
  });
  feedIndex = 0;
  renderFeed();
  renderPipeline({ allocation: { mode: "picked-taste", lambda: 0.55, sources: ["taste"] }, slices: {}, returned: feed.length }, "picked");
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
    try {
      const isbns = [...lovedBooks.keys()];
      const rows = await rpc("get_books_by_isbns", { isbns });
      lovedVecs = (rows || []).map((r) => toVec(r.combined_vec)).filter(Boolean);
    } catch (e) { console.warn("loved vecs", e); }
  }
  buildTasteFromOnboarding(genresArr, lovedVecs);
  state.total_swipes += lovedBooks.size;
  state.positive_signals += lovedBooks.size;
  // count onboarding books as seen-ish for coverage but not blacklisted
  lovedBooks.clear();
  $("apply-onboarding").disabled = true;
  updateGenreChips();
  renderChips();
  renderState(); renderClusters(); renderSignals();
  await fetchFeed();
}

async function sendSignal(isbn, signal, btn) {
  const book = feed.find((f) => f.isbn === isbn);
  const vec = book && book.vec ? book.vec : null;
  // snapshot for undo (deep copy clusters + disliked + shownCounts)
  histStack.push({
    snapshot: {
      has_taste: state.has_taste, total_swipes: state.total_swipes,
      positive_signals: state.positive_signals, tick: state.tick, signals: [...state.signals],
      clusters: state.clusters.map((c) => ({ ...c, centroid: c.centroid.slice() })),
      disliked: state.disliked ? { vec: state.disliked.vec.slice(), weight: state.disliked.weight } : null,
      nextId: state.nextId,
      shownCounts: { ...state.shownCounts },
      seen: new Set(state.seen),
      blacklist: new Set(state.blacklist),
    },
    isbn, signal,
  });
  state.total_swipes += 1;
  // tick managed inside apply* for taste/disliked; also bump for signal log
  const isPositive = signal === "like" || signal === "save" || signal === "tap";
  if (isPositive) state.positive_signals += 1;
  if (vec) {
    if (signal === "not_for_me") {
      applyDislikedSignal(vec);
    } else if (isPositive) {
      // like/save/tap all map to weight 1 (tap 0.5 in prod but browser treats as positive)
      applyTasteSignal(l2Normalize(vec.slice()));
    }
  } else {
    state.tick += 1;
  }
  state.seen.add(isbn);
  state.signals.push({ tick: state.tick, signal, isbn });
  if (signal === "not_for_me") state.blacklist.add(isbn);
  // update coverage
  state.coverage = Math.min(1, state.seen.size / TOTAL_CATALOG);
  // freshness shown count already updated on fetch; also bump on explicit signal
  state.shownCounts[isbn] = (state.shownCounts[isbn] || 0) + 1;
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
    state.disliked = s.disliked;
    state.nextId = s.nextId;
    state.shownCounts = s.shownCounts;
    state.seen = s.seen;
    state.blacklist = s.blacklist;
    state.coverage = Math.min(1, state.seen.size / TOTAL_CATALOG);
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
  feed = []; selectedGenres.clear(); lovedBooks.clear(); histStack = []; lastSeed = 0; feedIndex = 0;
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
  b.push(`<span class="badge warm">unified scoring + MMR</span>`);
  b.push(`<span class="badge warm">quality: ${QUALITY_TAG_STRICT}</span>`);
  $("state-badges").innerHTML = b.join("");
  renderSession();
}

function renderSession() {
  if (!state) return;
  const covPct = (state.coverage * 100).toFixed(2);
  const dislikedW = state.disliked ? state.disliked.weight.toFixed(2) : "—";
  const rows = [
    ["Coverage", `${covPct}%`],
    ["Signals", `${state.total_swipes}`],
    ["Positive", `${state.positive_signals}`],
    ["Clusters", `${state.clusters.length} / 5`],
    ["Disliked w", `${dislikedW}`],
    ["Lambda", `${(state.allocation && state.allocation.lambda != null) ? state.allocation.lambda : "—"}`],
  ];
  $("session-list").innerHTML = rows.map(([k, v]) =>
    `<div class="session-row"><span class="k">${k}</span><span class="v">${v}</span></div>`
  ).join("") +
  `<div class="session-tip">Coverage — seen / catalog. Lambda — MMR diversity (0.35 relevance → 0.85 explore). Disliked w — accumulated negative centroid weight (decay 0.90).</div>`;
}

function renderClusters() {
  const el = $("cluster-list");
  $("cluster-empty").style.display = state.clusters.length || state.disliked ? "none" : "";
  let html = state.clusters.map((c) => {
    const neg = c.weight < 0;
    const w = Math.min(Math.abs(c.weight) * 50, 100);
    return `<div class="cluster">
      <div class="row"><span><b class="${neg ? "neg" : ""}">${escapeHtml((c.label || "?").replace(/[\s/|+-]+$/, ""))}</b> · ${c.size} sig</span>
        <span>w=${c.weight.toFixed(2)} · #${c.id}</span></div>
      <div class="bar"><div class="${neg ? "neg" : ""}" style="width:${w}%"></div></div>
    </div>`;
  }).join("");
  if (state.disliked) {
    const dw = Math.min(state.disliked.weight * 20, 100);
    html += `<div class="cluster" style="opacity:0.85"><div class="row"><span><b class="neg">disliked</b> · not-for-me centroid</span><span>w=${state.disliked.weight.toFixed(2)}</span></div><div class="bar"><div class="neg" style="width:${dw}%"></div></div></div>`;
  }
  el.innerHTML = html;
}

function renderPipeline(stats, m) {
  const el = $("pipeline");
  if (!state) { el.innerHTML = ""; $("pipeline-empty").style.display = ""; return; }
  if (!stats || (stats.note === "no-taste" && m === "picked") || (!state.has_taste && m !== "search" && !stats.allocation)) {
    // keep helpful hint before first fetch
    if (!state.has_taste && !stats) {
      el.innerHTML = `<div class="hint">No taste profile yet. Pick genres and apply to build your taste.</div>`;
      $("pipeline-empty").style.display = "none";
      return;
    }
  }
  if (!stats) { el.innerHTML = ""; $("pipeline-empty").style.display = ""; return; }
  $("pipeline-empty").style.display = "none";
  const a = stats.allocation || {};
  let head = "";
  if (m === "fyp" || m === "disc") {
    const srcs = (a.sources || []).join(" + ") || (a.mode || "?");
    head = `<div class="slice"><h3>${escapeHtml(a.mode || "unified")}</h3><div class="kv"><span>sources: <b>${escapeHtml(srcs)}</b> · λ=${a.lambda ?? "?"} · cov=${a.coverage ?? "?"}</span><span>clusters: <b>${a.clusters_selected ?? "?"} / ${a.clusters_total ?? state.clusters.length}</b></span></div></div>`;
  } else if (m === "picked") {
    head = `<div class="slice"><h3>Picked for you</h3><div class="kv"><span>taste ANN → score → MMR</span><span>λ=0.55</span></div></div>`;
  }
  const slices = renderSliceStats(stats.slices, m, stats);
  const note = m === "picked" ? `Picked: ANN against your taste centroids, scored (w_taste 0.45) then MMR.`
                              : `Unified: each candidate scored (w_taste/w_genre/w_quality/w_fresh per source), then <b>MMR</b> reranked for diversity. Returned: <b>${stats.returned ?? "?"}</b> books.`;
  el.innerHTML = `${head}${slices}
    <div class="hint">${note}</div>`;
}

function renderSliceStats(slices, m, stats) {
  let out = "";
  const scoring = stats && stats.scoring ? `scored: <b>${stats.scoring.scored_total ?? "?"}</b> · pre-MMR: <b>${stats.scoring.pre_mmr ?? "?"}</b>` : "";
  if (m === "fyp" || m === "disc") {
    const t = (slices || {}).taste || {};
    const tr = (slices || {}).trending || {};
    const ex = (slices || {}).exploration || {};
    if (t.pool != null || t.ann_returned != null)
      out += sliceHtml("Taste pool · ANN nearest (scored w_taste 0.45)", `ann: <b>${t.ann_returned ?? 0}</b> · pool: <b>${t.pool ?? 0}</b>`, `picked: <b class="hot">${t.picked ?? 0}</b>`);
    if (tr.sampled != null || tr.pool != null)
      out += sliceHtml("Trending pool · popular random (scored w_quality 0.45)", `sampled: <b>${tr.sampled ?? 0}</b> · pool: <b>${tr.pool ?? 0}</b>`, `picked: <b class="hot">${tr.picked ?? 0}</b>`);
    if (ex.sampled != null)
      out += sliceHtml("Exploration pool · diverse random (scored w_fresh 0.55)", `sampled: <b>${ex.sampled ?? 0}</b> · pool: <b>${ex.pool ?? 0}</b>`, `picked: <b class="hot">${ex.picked ?? 0}</b>`);
    if (scoring) out += `<div class="slice"><h3>Scoring → MMR</h3><div class="kv"><span>${scoring}</span><span>λ=${(stats.allocation && stats.allocation.lambda) ?? "?"}</span></div></div>`;
  } else {
    for (const [key, label] of [["taste", "Taste"], ["trending", "Trending"], ["exploration", "Exploration"]]) {
      const s = (slices || {})[key];
      if (!s) continue;
      out += sliceHtml(label, `pool: <b>${s.pool ?? s.sampled ?? 0}</b>`, `picked: <b class="hot">${s.picked ?? 0}</b>`);
    }
    if (scoring) out += `<div class="slice"><h3>Scoring → MMR</h3><div class="kv"><span>${scoring}</span><span></span></div></div>`;
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
      ${f.source ? `<span class="slot-tag ${f.source}">${escapeHtml(f.source)}</span>` : (f.slice ? `<span class="slot-tag ${f.slice}">${escapeHtml(f.slice)}</span>` : "")}
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
  if (f.slice === "search" || f.source === "search") {
    const parts = ["<b>search</b>"];
    if (f.src) parts.push(`src: ${escapeHtml(f.src)}`);
    return parts.join(" · ");
  }
  const parts = [];
  const srcLabel = f.source || f.mmrSource || f.slice || "";
  if (srcLabel) parts.push(`<b>${escapeHtml(srcLabel)}</b>`);
  if (f.score != null) parts.push(`score=${f.score.toFixed(2)}`);
  if (f._comp) {
    // compact comp: t/q/f
    const c = f._comp;
    parts.push(`t=${c.t.toFixed(2)} q=${c.q.toFixed(2)} f=${c.f.toFixed(2)}`);
  } else if (f.dist != null) parts.push(`dist=${f.dist.toFixed(2)}`);
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
  const genresArr = parseGenres(book.genres);
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
