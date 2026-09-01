import os
import sys
from pathlib import Path

from supabase import create_client, ClientOptions

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    d = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            d[k.strip()] = v.strip().strip('"').strip("'")
    return d


env = load(ROOT / ".env")
url = os.environ.get("DEMO_SUPABASE_URL") or env.get("DEMO_SUPABASE_URL") or "https://hkdtpwbmgmnxhykhgqvb.supabase.co"
key = os.environ.get("DEMO_SUPABASE_SERVICE_ROLE_KEY") or env.get("DEMO_SUPABASE_SERVICE_ROLE_KEY")

if not key or key.startswith("eyJ.REPLACE"):
    sys.exit("no demo service_role key in .env")

c = create_client(url, key, options=ClientOptions(postgrest_client_timeout=1200))
print("calling _build_bq_hnsw ...", flush=True)
try:
    r = c.rpc("_build_bq_hnsw").execute()
    print("DONE", r.data, flush=True)
except Exception as e:  # noqa: BLE001
    print("ERROR", repr(e), flush=True)
    sys.exit(1)
