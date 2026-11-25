"""Colab launcher, live monitor & auto-download helper for MeguvizCAC v0.4.0 R3 (A100-80GB)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_r3_in_colab(poll_seconds: int = 90) -> dict:
    drive_out = Path("/content/drive/MyDrive/MeguvizCAC/r3_v040")
    if not drive_out.is_dir():
        raise RuntimeError("Google Drive 未挂载！请先运行 Cell 1 授权并挂载 /content/drive。")

    driver_sh = Path("/content/MeguvizCAC/meguviz_cac/run_r3_a100_80g.sh")
    for req in (
        "/content/data/v04-merged-1m4-r3.jsonl.gz",
        "/content/data/co-teacher-gate/v04-gateguard-pairs.jsonl",
        "/content/data/v04_semantic_intent_cache.npz",
        "/content/MeguvizCAC/meguviz_cac/train_v04_dualloop.py",
        str(driver_sh),
    ):
        if not Path(req).is_file():
            raise FileNotFoundError(f"前置资产缺失，请先运行 Cell 1 & Cell 2: {req}")

    os.chmod(driver_sh, 0o755)
    subprocess.run(["bash", "-n", str(driver_sh)], check=True)

    pid_file = Path("/content/r3.pid")
    driver_log = Path("/content/log_r3_driver.txt")
    train_log = Path("/content/log_r3_full.txt")

    if pid_file.is_file() and _pid_alive(int(pid_file.read_text(encoding="utf-8").strip())):
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        print(f"R3 driver already running, re-attaching pid={pid}", flush=True)
    else:
        Path("/content/r3_done.json").unlink(missing_ok=True)
        proc = subprocess.Popen(
            f"setsid nohup bash {driver_sh} > {driver_log} 2>&1 < /dev/null & echo $!",
            shell=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        pid = int(proc.communicate()[0].strip())
        pid_file.write_text(str(pid), encoding="utf-8")
        time.sleep(5)
        print(f"Launched R3 A100-80GB background driver pid={pid}", flush=True)

    tick = 0
    while _pid_alive(pid):
        tick += 1
        now = time.strftime("%H:%M:%S")
        drv_tail = ""
        if driver_log.is_file():
            lines = driver_log.read_text(encoding="utf-8", errors="replace").splitlines()
            drv_tail = " | ".join(lines[-2:]) if lines else ""
        stage_line = ""
        if train_log.is_file():
            for ln in reversed(train_log.read_text(encoding="utf-8", errors="replace").splitlines()):
                if ln.startswith(("[Stage", "[Config")):
                    stage_line = ln[:170]
                    break
        gpu_stat = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
            text=True,
        ).strip()
        ram_stat = subprocess.check_output(
            "free -g | awk '/Mem:/{print \"RAM \" $3 \"G/\" $2 \"G\"}'", shell=True, text=True
        ).strip()
        print(f"=== {now} (tick {tick}) === {gpu_stat} | {ram_stat}", flush=True)
        if drv_tail:
            print(f"  driver: {drv_tail}", flush=True)
        if stage_line:
            print(f"  train : {stage_line}", flush=True)
        if tick % 5 == 0 and train_log.is_file():
            try:
                shutil.copy2(train_log, drive_out / "log_r3_full_live.txt")
            except Exception as exc:
                print(f"  [warn] live log sync: {exc}", flush=True)
        time.sleep(poll_seconds)

    print("\n=== R3 DRIVER EXITED ===", flush=True)
    if driver_log.is_file():
        print(driver_log.read_text(encoding="utf-8", errors="replace"), flush=True)

    done_file = Path("/content/r3_done.json")
    if not done_file.is_file():
        raise RuntimeError(
            "R3 训练未生成完成标记 r3_done.json，请检查上方 log_r3_driver.txt 与 /content/log_r3_full.txt！"
        )

    info = json.loads(done_file.read_text(encoding="utf-8"))
    print("R3_DONE_SUMMARY:", json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    full_zip = Path(info["full_zip"])
    if full_zip.is_file():
        print(f"Triggering automatic browser download: {full_zip.name}", flush=True)
        from google.colab import files

        files.download(str(full_zip))
    return info
