"""Exit 1 if the Claude subscription's 5-hour session or weekly usage is at or above the limits.
Reads `claude -p /usage` (a local command: no model call, no cost). Exit 2 if it cannot be read."""
import json, re, subprocess, sys, time

SESSION, WEEK = float(sys.argv[1]), float(sys.argv[2])
try:
    out = subprocess.run(["claude", "-p", "/usage", "--output-format", "json"], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=120, cwd="/tmp").stdout
    text = json.loads(out)["result"]
    session = float(re.search(r"Current session: (\d+(?:\.\d+)?)% used", text).group(1))
    week = float(re.search(r"Current week \(all models\): (\d+(?:\.\d+)?)% used", text).group(1))
except Exception as error:
    print(json.dumps({"time": time.strftime("%H:%M:%S"), "error": str(error)[:200]}))
    sys.exit(2)
print(json.dumps({"time": time.strftime("%H:%M:%S"), "session_pct": session, "week_pct": week}))
sys.exit(1 if session >= SESSION or week >= WEEK else 0)
