#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

gpu=${1:-A100}
case "$gpu" in
    A100|H100|L4|T4|G4) ;;
    *) printf 'Usage: bash scripts/run_colab_smoke.sh [A100|H100|L4|T4|G4] [--compare-caps]\n' >&2; exit 2 ;;
esac

upload_files=(fixed8.py verifier.py smoke_fixed8.py validation_smoke.jsonl)
case "${2:-}" in
    '') driver=smoke_fixed8.py; summary=update.json ;;
    --compare-caps)
        driver=compare_fixed8_caps.py
        summary=cap_comparison.json
        upload_files+=("$driver")
        ;;
    *) printf 'Unknown mode: %s\n' "$2" >&2; exit 2 ;;
esac

missing_inputs=0
for file in "${upload_files[@]}" requirements-colab.txt; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing required input: %s\n' "$file" >&2
        missing_inputs=1
    fi
done
if (( missing_inputs )); then
    exit 1
fi

python3 -B -m unittest discover -s tests -v
if ! command -v colab >/dev/null; then
    printf 'Install the official Colab CLI with uv tool install google-colab-cli.\n' >&2
    exit 1
fi

auth_provider=${COLAB_AUTH_PROVIDER:-oauth2}
colab_cli() {
    colab "--auth=$auth_provider" "$@"
}

mkdir -p artifacts
run_dir=$(mktemp -d "$PWD/artifacts/fixed8.XXXXXXXX")
session="fixed8-${run_dir##*.}"
printf 'Colab session %s\nLocal logs %s\n' "$session" "$run_dir"
colab_cli version | tee "$run_dir/cli-version.txt"
colab_cli sessions >/dev/null
colab_cli new -s "$session" --gpu "$gpu"

cleanup() {
    local status=$?
    trap - EXIT
    if ! colab_cli log -s "$session" -o "$run_dir/session.jsonl"; then
        printf 'Could not export session history for %s.\n' "$session" >&2
    fi
    if ! colab_cli stop -s "$session"; then
        printf 'Could not stop %s. Retry colab --auth=%s stop -s %s.\n' "$session" "$auth_provider" "$session" >&2
        if (( status == 0 )); then
            status=1
        fi
    fi
    printf 'Local logs %s\n' "$run_dir"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

colab_cli status -s "$session" 2>&1 | tee "$run_dir/status.txt"
colab_cli exec -s "$session" <<'PY' 2>&1 | tee "$run_dir/gpu.txt"
import subprocess

import torch

print(subprocess.check_output(["nvidia-smi"], text=True))
assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print(torch.ones(1, device="cuda"))
print("FIXED8_CUDA_OK")
PY
if ! grep -Fxq 'FIXED8_CUDA_OK' "$run_dir/gpu.txt"; then
    printf 'The CUDA preflight did not complete. See gpu.txt.\n' >&2
    exit 1
fi

colab_cli install -s "$session" -r requirements-colab.txt 2>&1 | tee "$run_dir/install.txt"
for file in "${upload_files[@]}"; do
    colab_cli upload -s "$session" "$file" "/content/$file"
done

smoke_status=0
colab_cli exec -s "$session" --timeout 600 <<PY 2>&1 | tee "$run_dir/console.txt" || smoke_status=$?
import subprocess
import sys

with subprocess.Popen(
    [sys.executable, "-u", "$driver"],
    cwd="/content",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise SystemExit(process.returncode)
print("FIXED8_DRIVER_OK")
PY

download_status=0
for file in rollouts.jsonl "$summary"; do
    if ! colab_cli download -s "$session" "/content/$file" "$run_dir/$file"; then
        printf 'Could not retrieve %s.\n' "$file" >&2
        download_status=1
    elif [[ ! -s "$run_dir/$file" ]]; then
        printf 'The downloaded %s is empty or missing.\n' "$file" >&2
        download_status=1
    fi
done

if (( smoke_status != 0 )); then
    exit "$smoke_status"
fi
if ! grep -Fxq 'FIXED8_DRIVER_OK' "$run_dir/console.txt"; then
    printf 'The smoke driver did not complete. See console.txt.\n' >&2
    exit 1
fi
exit "$download_status"
