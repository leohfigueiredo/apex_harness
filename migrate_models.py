#!/usr/bin/env python3
"""
Migrate AI Models from Windows NTFS to Crucial P3 SSD with real-time status reporting.
"""
import os
import sys
import re
import json
import time
import subprocess
from pathlib import Path

SRC = "/run/media/leonardo/Windows/AIModels/"
DST = "/run/media/leonardo/C69ADBDF9ADBCA57/AIModels/"
STATUS_FILE = "/home/leonardo/.apex_sessions/migration_status.json"
STATUS_TMP = "/home/leonardo/.apex_sessions/migration_status.json.tmp"
LOG_FILE = "/home/leonardo/.apex_sessions/migration.log"

def update_status(data):
    try:
        with open(STATUS_TMP, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(STATUS_TMP, STATUS_FILE)
    except Exception as e:
        print(f"Error saving status: {e}", file=sys.stderr)

def fix_lmstudio_symlinks(dst_dir):
    links_dir = Path(dst_dir) / "_LMStudio_Links"
    if not links_dir.exists():
        return
    fixed = 0
    for root, _, files in os.walk(links_dir):
        for f in files:
            full_path = Path(root) / f
            if full_path.is_symlink():
                target = os.readlink(full_path)
                if "/run/media/leonardo/Windows/AIModels" in target:
                    new_target = target.replace("/run/media/leonardo/Windows/AIModels", "/run/media/leonardo/C69ADBDF9ADBCA57/AIModels")
                    try:
                        full_path.unlink()
                        full_path.symlink_to(new_target)
                        fixed += 1
                    except Exception as err:
                        print(f"Failed to update symlink {full_path}: {err}", file=sys.stderr)
    print(f"Updated {fixed} LM Studio symlinks to target new SSD location.")

def main():
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    os.makedirs(DST, exist_ok=True)
    
    start_time = time.time()
    initial_status = {
        "status": "starting",
        "percent": 0,
        "speed": "0 MB/s",
        "eta": "calculating",
        "bytes_transferred": 0,
        "elapsed_seconds": 0,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": SRC,
        "destination": DST
    }
    update_status(initial_status)

    cmd = [
        "rsync",
        "-rltvh",
        "--no-perms",
        "--no-owner",
        "--no-group",
        "--info=progress2",
        SRC,
        DST
    ]

    log_fp = open(LOG_FILE, "w", buffering=1)
    log_fp.write(f"=== Migration started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_fp.write(f"Command: {' '.join(cmd)}\n\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    progress_regex = re.compile(r"(\d+(?:,\d+)*|\d+(?:\.\d+)?[KMGTP]?)\s+(\d+)%\s+([\d\.]+[KMGTP]?B/s)\s+([0-9:]+)")

    last_update = 0
    buf = ""

    while True:
        char = proc.stdout.read(1)
        if not char:
            break
        if char in ('\r', '\n'):
            line = buf.strip()
            buf = ""
            if line:
                log_fp.write(line + "\n")
                match = progress_regex.search(line)
                now = time.time()
                if match and (now - last_update >= 1.0):
                    last_update = now
                    transferred_str = match.group(1)
                    pct = int(match.group(2))
                    speed = match.group(3)
                    eta = match.group(4)
                    
                    status_payload = {
                        "status": "transferring",
                        "percent": pct,
                        "speed": speed,
                        "eta": eta,
                        "transferred_raw": transferred_str,
                        "elapsed_seconds": int(now - start_time),
                        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                        "source": SRC,
                        "destination": DST
                    }
                    update_status(status_payload)
        else:
            buf += char

    proc.wait()
    log_fp.close()

    if proc.returncode == 0:
        print("Rsync completed successfully. Updating LM Studio symlinks...")
        fix_lmstudio_symlinks(DST)

        end_status = {
            "status": "completed",
            "percent": 100,
            "speed": "0 MB/s",
            "eta": "0:00:00",
            "elapsed_seconds": int(time.time() - start_time),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "source": SRC,
            "destination": DST
        }
        update_status(end_status)
        print("Migration finished with returncode 0.")
    else:
        err_status = {
            "status": "failed",
            "returncode": proc.returncode,
            "elapsed_seconds": int(time.time() - start_time),
            "source": SRC,
            "destination": DST
        }
        update_status(err_status)
        print(f"Migration failed with returncode {proc.returncode}.", file=sys.stderr)
        sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
