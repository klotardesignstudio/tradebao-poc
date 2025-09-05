#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main scheduler: runs update_enriched and paper_trading every 4 hours.

- Uses the same Python interpreter to invoke the scripts
- Logs start/end times and handles errors without crashing the loop
"""

import os
import sys
import time
import subprocess
import argparse
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script: str, args: list[str] | None = None) -> int:
    cmd = [sys.executable, os.path.join(BASE_DIR, script)]
    if args:
        cmd.extend(args)
    print(f"\n[{datetime.utcnow().isoformat()}Z] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        print(f"[{datetime.utcnow().isoformat()}Z] Finished: {script} (exit={result.returncode})")
        return result.returncode
    except Exception as e:
        print(f"[{datetime.utcnow().isoformat()}Z] ERROR running {script}: {e}")
        return 1


def cycle(interval_seconds: int, once: bool) -> None:
    while True:
        # 1) Update enriched data (avoid gaps)
        run_script('update_enriched.py', ['--interval', '4h', '--limit', '600'])

        # 2) Run paper trading (simulated execution)
        run_script('paper_trading.py')

        if once:
            break

        print(f"[{datetime.utcnow().isoformat()}Z] Sleeping for {interval_seconds // 3600}h...")
        try:
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("Interrupted during sleep. Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(description='TradeBao main loop: update enriched + paper trading every 4 hours')
    parser.add_argument('--interval-hours', type=int, default=4, help='Interval between runs in hours (default: 4)')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()

    interval_seconds = max(1, args.interval_hours) * 3600

    try:
        cycle(interval_seconds, args.once)
    except KeyboardInterrupt:
        print("Interrupted. Exiting.")


if __name__ == '__main__':
    main()


