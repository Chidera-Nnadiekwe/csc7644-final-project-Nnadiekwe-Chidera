"""
gc_watcher.py
-------------
File-system watcher that monitors a directory for new GC export CSV files
and automatically ingests them as experimental results, eliminating the need
for the lab chemist to manually type values into the terminal.

How it works
------------
1.  The agent calls ``wait_for_gc_result(watch_dir, timeout_s)`` instead of
    prompting the user.
2.  The watcher polls ``watch_dir`` for any new ``.csv`` file that was not
    present when the call started (or that has a modification time newer than
    the call start time).
3.  When a new CSV is detected, it is parsed via ``result_parser.parse_from_gc_csv``
    and the structured outcome dict is returned to the controller.
4.  If ``timeout_s`` elapses with no new file, the watcher falls back to
    interactive manual entry so the agent loop is never permanently blocked.

Usage (integrated into agent_controller.py through --ingest-mode gc-watch):
    outcomes = wait_for_gc_result(watch_dir="data/gc_drops/", timeout_s=300)

Expected CSV format (two-column, comma-separated, no header required):
    substrate,1500.0
    linear_aldehyde,4800.0
    branch_aldehyde,1200.0

See result_parser.parse_from_gc_csv for full format documentation.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add src/ to path so result_parser is importable when called standalone
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from result_parser import parse_from_gc_csv, parse_experimental_result

# Constants and defaults

POLL_INTERVAL_S  = 2.0    # seconds between directory scans
DEFAULT_TIMEOUT  = 300    # 5-minute default timeout before falling back to manual
SUPPORTED_EXTS   = {".csv", ".txt"}   # extensions considered GC drop files


# Public API 

def wait_for_gc_result(
    watch_dir: str  = "data/gc_drops",
    timeout_s: int  = DEFAULT_TIMEOUT,
    move_to_done: bool = True,
) -> dict:
    """Block until a new GC CSV appears in ``watch_dir``, then parse and return it.

    Parameters
    ----------
    watch_dir : str
        Directory to watch.  Created automatically if it does not exist.
    timeout_s : int
        Seconds to wait before falling back to interactive manual entry.
    move_to_done : bool
        If True, move the consumed file to ``watch_dir/done/`` after parsing,
        so it is not re-ingested on the next call.

    Returns
    -------
    dict
        Validated outcome dict: ``{conversion_pct, l_b_ratio, ton, notes}``.
    """
    watch_path = Path(watch_dir)
    watch_path.mkdir(parents=True, exist_ok=True)
    done_path = watch_path / "done"
    done_path.mkdir(exist_ok=True)

    # Snapshot of files that exist *before* this call starts
    existing = _snapshot(watch_path)
    start    = time.monotonic()

    print(f"\n[GC-WATCH] Waiting for a new GC CSV in '{watch_path.resolve()}' ...")
    print(f"           (timeout in {timeout_s}s — drop a CSV file or press Ctrl+C to enter manually)\n")

    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= timeout_s:
                print(f"\n[GC-WATCH] Timeout after {timeout_s}s. Falling back to manual entry.")
                return _manual_fallback()

            current  = _snapshot(watch_path)
            new_files = current - existing

            if new_files:
                # Sort by modification time; take the oldest new file first
                chosen = sorted(
                    new_files,
                    key=lambda p: p.stat().st_mtime
                )[0]
                print(f"[GC-WATCH] New file detected: '{chosen.name}'")
                result = _parse_with_retry(chosen)
                if result is not None:
                    if move_to_done:
                        dest = done_path / chosen.name
                        chosen.rename(dest)
                        print(f"[GC-WATCH] File moved to '{dest}'.")
                    return result
                else:
                    print(f"[GC-WATCH] Could not parse '{chosen.name}'. Waiting for another file...")
                    # Add to existing so that bad file is not retried
                    existing.add(chosen)

            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        print("\n[GC-WATCH] Interrupted. Switching to manual entry.")
        return _manual_fallback()


def list_pending_files(watch_dir: str = "data/gc_drops") -> list:
    """Return a list of CSV/TXT files currently sitting in the watch directory.

    Useful for diagnostics and testing.
    """
    watch_path = Path(watch_dir)
    if not watch_path.exists():
        return []
    return [
        p for p in sorted(watch_path.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]


# Internal helpers 
# Define as internal (underscore prefix) since these are not intended to be used outside of this module, and their behavior is not guaranteed to be stable.
def _snapshot(watch_path: Path) -> set:
    """Return the set of CSV/TXT Path objects currently in ``watch_path``."""
    return {
        p for p in watch_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    }

# Define as internal since this is a helper function used by wait_for_gc_result and not intended for external use.
def _parse_with_retry(filepath: Path, max_retries: int = 3) -> Optional[dict]:
    """Attempt to parse a GC CSV, retrying briefly in case the file is still
    being written by the instrument export software."""
    for attempt in range(1, max_retries + 1):
        result = parse_from_gc_csv(str(filepath))
        if result is not None:
            return result
        if attempt < max_retries:
            print(f"[GC-WATCH] Parse attempt {attempt} failed; retrying in {POLL_INTERVAL_S}s ...")
            time.sleep(POLL_INTERVAL_S)
    return None

# Define as internal since this is a fallback function used by wait_for_gc_result and not intended for external use.
def _manual_fallback() -> dict:
    """Interactive CLI prompt used when the watcher times out or is interrupted."""
    print("\n[GC-WATCH] Manual entry mode.")

    def prompt_float(label: str, default: float) -> float:
        raw = input(f"  {label} [{default}]: ").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    conversion = prompt_float("Conversion (%)", 45.0)
    l_b_ratio  = prompt_float("L:B Ratio",       2.5)
    ton        = prompt_float("TON",             120.0)
    notes      = input("  Notes (optional): ").strip()
    return parse_experimental_result(conversion, l_b_ratio, ton, notes)


# CLI for manual testing

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GC file-watcher: drop a GC CSV to trigger automatic parsing."
    )
    parser.add_argument(
        "--watch-dir", default="data/gc_drops",
        help="Directory to monitor (default: data/gc_drops)"
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Seconds before falling back to manual entry (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--no-move", action="store_true",
        help="Do not move processed files to the done/ subdirectory"
    )
    args = parser.parse_args()

    result = wait_for_gc_result(
        watch_dir=args.watch_dir,
        timeout_s=args.timeout,
        move_to_done=not args.no_move,
    )
    print("\n[GC-WATCH] Parsed result:")
    import json
    print(json.dumps(result, indent=2))
