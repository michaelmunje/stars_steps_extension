"""Judge usage so far: sum of `claude -p` cost and tokens over every judge_transcript.json under the given dirs."""
import json, sys
from pathlib import Path
total = {"calls": 0, "cost_usd": 0.0, "input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
for root in sys.argv[1:]:
    for p in Path(root).rglob("judge_transcript*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        u = r.get("usage", {})
        total["calls"] += 1
        total["cost_usd"] += float(r.get("total_cost_usd") or 0)
        total["input"] += u.get("input_tokens", 0); total["output"] += u.get("output_tokens", 0)
        total["cache_read"] += u.get("cache_read_input_tokens", 0); total["cache_write"] += u.get("cache_creation_input_tokens", 0)
print(json.dumps(total))
