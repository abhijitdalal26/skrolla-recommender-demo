"""Copy the live production book catalog (book_embeddings) + genre anchors into the
isolated portfolio demo Supabase project.

Source  = production project (rhljzgnybukfqmdujqfn) — this is the SAME data the
          Android app serves (74,732 rows after the NYT reembed + bad-cover cleanup).
          Read via service_role.
Target  = demo project (hkdtpwbmgmnxhykhgqvb) — a fresh, isolated project.
          Written via service_role.

Run regularly to refresh the demo; the copy is idempotent (upsert on isbn13 / genre).

Usage:
    python copy_prod_to_demo.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from supabase import create_client, ClientOptions

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader: exports KEY=VALUE lines into the environment (no override)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load secrets from a gitignored .env at the project root, if present.
_load_dotenv(ROOT / ".env")

# Source keys (production) — read from local gitignored mirror, else env.
_PROD_URL = os.environ.get("SUPABASE_PROJECT_URL") or (
    ROOT / ".." / "Skrolla" / "supabase" / ".project_url").read_text().strip()
_PROD_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or (
    ROOT / ".." / "Skrolla" / "supabase" / ".service_role_key").read_text().strip()

# Target keys (demo) — read from .env (or env), else a gitignored fallback file.
# NEVER commit these to the repo.
DEMO_URL = os.environ.get("DEMO_SUPABASE_URL") or "https://hkdtpwbmgmnxhykhgqvb.supabase.co"
DEMO_KEY = os.environ.get("DEMO_SUPABASE_SERVICE_ROLE_KEY") or (
    ROOT / ".demo_service_role_key").read_text().strip()

EMBED_BATCH = 400      # rows per upsert batch (768-d halfvec payloads)
PAGE = 500             # source rows per read query (large payloads -> keep small to dodge statement_timeout)
MAX_ATTEMPTS = 6       # retries per source page read on transient timeout
CURSOR_FILE = ROOT / ".copy_cursor"
META_COLS = [
    "isbn13", "title", "author", "genres", "description", "pub_year",
    "avg_rating", "ratings_count", "source_count", "cover_file", "src",
    "canonical_genres", "is_nyt_bestseller",
]
VEC_COL = "combined_vec"


def copy_book_embeddings(src, dst):
    # Resumable: pick up where the last successful page ended (covers survive a
    # timeout mid-copy; upsert on isbn13 is idempotent so re-running is safe).
    cursor = 0
    if CURSOR_FILE.exists():
        cur = CURSOR_FILE.read_text().strip()
        if cur and cur.isdigit():
            cursor = int(cur)
        print(f"  resuming from id > {cursor}", flush=True)

    n = 0
    while True:
        # Read one page from the source. Retry on transient statement_timeout
        # (prod has a strict statement_timeout and 768-d payloads are heavy).
        rows = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                q = src.table("book_embeddings").select("*").order("id").limit(PAGE)
                if cursor:
                    q = q.gt("id", cursor)
                res = q.execute()
                rows = res.data
                break
            except Exception as exc:  # noqa: BLE001 - retry transient timeouts
                if attempt == MAX_ATTEMPTS:
                    print(f"  giving up at cursor {cursor}: {exc}", flush=True)
                    raise
                time.sleep(2)

        if not rows:
            break

        for start in range(0, len(rows), EMBED_BATCH):
            batch = rows[start:start + EMBED_BATCH]
            payloads = [
                {c: r.get(c) for c in META_COLS} | {VEC_COL: r[VEC_COL]}
                for r in batch
            ]
            dst.table("book_embeddings").upsert(
                payloads, on_conflict="isbn13"
            ).execute()

        cursor = rows[-1]["id"]
        CURSOR_FILE.write_text(str(cursor))
        n += len(rows)
        print(f"  {n} (cursor {cursor})", flush=True)
        if len(rows) < PAGE:
            break
    print(f"Done: {n} rows")
    CURSOR_FILE.unlink(missing_ok=True)


def copy_genre_anchors(src, dst):
    rows = src.table("onboarding_genre_anchors").select("*").execute().data
    print(f"Copying {len(rows)} genre anchors...")
    # generated_at is a timestamp; re-derive on insert. Drop it to let default now() apply.
    for r in rows:
        r.pop("generated_at", None)
    if rows:
        dst.table("onboarding_genre_anchors").upsert(rows, on_conflict="genre").execute()
    print("Genre anchors done.")


def main():
    if not DEMO_KEY or DEMO_KEY.startswith("eyJ.REPLACE"):
        raise SystemExit(
            "Set DEMO_SUPABASE_SERVICE_ROLE_KEY or write the demo service_role key to "
            f"{ROOT / '.demo_service_role_key'}"
        )
    src = create_client(_PROD_URL, _PROD_KEY, options=ClientOptions(postgrest_client_timeout=300))
    dst = create_client(DEMO_URL, DEMO_KEY, options=ClientOptions(postgrest_client_timeout=300))
    copy_book_embeddings(src, dst)
    copy_genre_anchors(src, dst)


if __name__ == "__main__":
    main()
