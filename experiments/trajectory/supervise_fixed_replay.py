"""Restart a resumable replay if needed, then generate its result report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--max-restarts", default=5, type=int)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.run_dir / "run.log"
    previous_completed = -1
    no_progress = 0
    for attempt in range(args.max_restarts + 1):
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_fixed_replay.py")),
            "--config", str(args.config),
            "--run-dir", str(args.run_dir),
            "--model-path", str(args.model_path),
            "--resume",
        ]
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
        completion_path = args.run_dir / "completion.json"
        completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
        completed = completion.get("completed_requests", 0)
        planned = completion.get("planned_requests")
        if planned is not None and completed == planned:
            break
        no_progress = no_progress + 1 if completed == previous_completed else 0
        previous_completed = completed
        if attempt == args.max_restarts or no_progress >= 2:
            raise RuntimeError(f"Replay incomplete after retries: {completed}/{planned}; last return code {return_code}")
        print(f"Restarting incomplete replay in 15 seconds ({completed}/{planned})", flush=True)
        time.sleep(15)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("analyze_fixed_replay.py")), "--run-dir", str(args.run_dir)],
        check=True,
    )


if __name__ == "__main__":
    main()
