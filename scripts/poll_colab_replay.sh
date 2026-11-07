# Sourced by run_colab_smoke.sh after CUDA, upload, and provenance checks.
# The remote collector outlives a console connection. File polls recover progress.

colab_cli exec -s "$session" --timeout 60 <<PY 2>&1 | tee "$run_dir/launch.txt"
import subprocess
import sys
import json
from pathlib import Path
import threading
import zipfile

command = [sys.executable, "-u", "collect_replay.py", "--plan", "config.json", "--split", "$replay_split"]
if "$replay_split" == "held_out":
    command.extend(["--frozen", "frozen.json"])
limit = json.loads(Path("/content/config.json").read_text()).get("collection_wall_seconds", 4200)
if Path("/content/resume.zip").exists():
    with zipfile.ZipFile("/content/resume.zip") as bundle:
        bundle.extractall("/content/resume")
    previous = json.loads(Path("/content/resume/collection.json").read_text())
    limit -= previous["elapsed_seconds"] + 120
    command.extend(["--resume-dir", "/content/resume"])
with open("/content/collector-console.txt", "w") as console:
    process = subprocess.Popen(command, cwd="/content", stdout=console, stderr=subprocess.STDOUT,
                               start_new_session=True)
print(f"REPLAY_LAUNCHED pid={process.pid}", flush=True)
watchdog = threading.Timer(max(0, limit), process.terminate)
watchdog.daemon = True
watchdog.start()
PY

poll_started=$SECONDS
poll_limit=$(python3 -B - "$run_dir/config.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("collection_wall_seconds", 4200) + 180)
PY
)
last_completed=-1
poll_failures=0
while (( SECONDS - poll_started <= poll_limit )); do
    if ! colab_cli download -s "$session" /content/collection.json "$run_dir/collection.current.json" >>"$run_dir/polling.txt" 2>&1; then
        poll_failures=$((poll_failures + 1))
        if (( poll_failures >= 3 )); then
            colab_cli sessions >>"$run_dir/recovery.txt" 2>&1 || true
            driver_status=1
            break
        fi
        sleep 10
        continue
    fi
    poll_failures=0
    read -r collection_status completed < <(python3 -B - "$run_dir/collection.current.json" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1]))
print(metrics["status"], metrics["completions"])
PY
)
    if (( completed > last_completed )) || [[ "$collection_status" != running ]]; then
        if ! colab_cli download -s "$session" /content/rollouts.jsonl "$run_dir/rollouts.download.jsonl" >>"$run_dir/polling.txt" 2>&1; then
            driver_status=1
            break
        fi
        python3 -B - "$run_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
metrics = json.loads((directory / "collection.current.json").read_text())
lines = (directory / "rollouts.download.jsonl").read_bytes().splitlines()
if len(lines) < metrics["completions"]:
    raise RuntimeError("Downloaded raw log is shorter than its summary")
raw = b"\n".join(lines[:metrics["completions"]]) + (b"\n" if metrics["completions"] else b"")
rows = [json.loads(line) for line in lines[:metrics["completions"]]]
if sum(row["generated_tokens"] for row in rows) != metrics["generated_tokens"]:
    raise RuntimeError("Downloaded token counts do not reconcile")
if metrics["status"] == "passed" and hashlib.sha256(raw).hexdigest() != metrics["cache_sha256"]:
    raise RuntimeError("Final cache checksum mismatch")
for stem in ("collection", f"step-{metrics['completions']}.collection"):
    target = directory / f"{stem}.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(metrics, indent=2) + "\n")
    temp.replace(target)
for stem in ("rollouts", f"step-{metrics['completions']}.rollouts"):
    target = directory / f"{stem}.jsonl"
    temp = target.with_suffix(".tmp")
    temp.write_bytes(raw)
    temp.replace(target)
print(f"Saved {metrics['completions']} completions, {metrics['generated_tokens']} generated tokens, status={metrics['status']}", flush=True)
PY
        colab_cli download -s "$session" /content/collector-console.txt "$run_dir/console.txt" >>"$run_dir/polling.txt" 2>&1 || true
        last_completed=$completed
    fi
    if [[ "$collection_status" == passed ]]; then
        break
    fi
    if [[ "$collection_status" == failed ]]; then
        driver_status=1
        break
    fi
    sleep 20
done
if [[ "${collection_status:-}" != passed ]]; then
    driver_status=1
fi
