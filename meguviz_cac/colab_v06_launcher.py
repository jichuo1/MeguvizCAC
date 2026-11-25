"""Thin Colab launcher & live watchdog for MeguvizCAC v0.6.0-HyperLoop (G4 / H100 / A100 / L4 / T4)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def run_v06_in_colab(
    repo_dir: str = "/content/MeguvizCAC",
    drive_dir: str = "/content/drive/MyDrive/MeguvizCAC/v060_hyperloop",
    poll_seconds: int = 90,
    sync_every_ticks: int = 6,
) -> dict:
    """Launch detached v0.6.0-HyperLoop GPU driver, monitor GPU/logs, sync logs to Drive, and trigger download."""
    repo = Path(repo_dir)
    drv = Path(drive_dir)
    drv.mkdir(parents=True, exist_ok=True)

    driver_sh = repo / "meguviz_cac" / "run_v06_gpu.sh"
    assert driver_sh.is_file(), f"Missing {driver_sh}"

    log_driver = Path("/content/log_v06_driver.txt")
    log_full = Path("/content/log_v06_full.txt")
    done_json = Path("/content/v06_done.json")
    for p in (log_driver, log_full, done_json):
        p.unlink(missing_ok=True)

    env = dict(os.environ)
    env["REPO_DIR"] = str(repo)
    env["DRIVE_DIR"] = str(drv)

    # Note: execute via `bash driver_sh` directly (no os.chmod) so git working tree stays 100% clean!
    proc = subprocess.Popen(
        ["bash", str(driver_sh)],
        cwd=str(repo),
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"LAUNCHED detached v0.6.0-HyperLoop driver pid={proc.pid} | drive_dir={drv}", flush=True)

    def _tail(path: Path, n: int = 2) -> str:
        if not path.is_file():
            return "(waiting)"
        lines = [ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        return " | ".join(lines[-n:]) if lines else "(empty)"

    def _gpu_line() -> str:
        try:
            g = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
                text=True,
            ).strip()
        except Exception:
            g = "nvidia-smi N/A"
        try:
            r = subprocess.check_output(["free", "-h"], text=True).splitlines()[1].split()
            ram = f"RAM {r[2]}/{r[1]}"
        except Exception:
            ram = "RAM N/A"
        return f"{g} | {ram}"

    tick = 0
    while proc.poll() is None:
        time.sleep(poll_seconds)
        tick += 1
        print(f"=== {time.strftime('%H:%M:%S')} (tick {tick}) === {_gpu_line()}", flush=True)
        print(f"  driver: {_tail(log_driver, 2)}", flush=True)
        print(f"  train : {_tail(log_full, 1)[:220]}", flush=True)
        if tick % sync_every_ticks == 0:
            for lp in (log_driver, log_full):
                if lp.is_file():
                    shutil.copy2(lp, drv / lp.name)

    rc = proc.returncode
    print(f"DRIVER_EXITED rc={rc}", flush=True)
    if log_driver.is_file():
        print("\n".join(log_driver.read_text(encoding="utf-8", errors="ignore").splitlines()[-45:]), flush=True)
    assert rc == 0 and done_json.is_file(), f"v0.6.0 driver failed (rc={rc})! Check {log_full}"
    info = json.loads(done_json.read_text(encoding="utf-8"))
    print("V06_DONE_SUMMARY:", json.dumps(info, ensure_ascii=False), flush=True)

    try:
        from google.colab import files  # type: ignore

        files.download(info["full_zip"])
        print(f"AUTO_DOWNLOAD_TRIGGERED: {info['full_zip']}", flush=True)
    except Exception as exc:
        print(f"AUTO_DOWNLOAD_SKIPPED ({exc}) — artifact safely on Drive: {info['drive_zip']}", flush=True)
    return info
