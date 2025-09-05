#!/usr/bin/env python3
"""
Utility to rename files under data/raw by removing a given prefix from filenames.

Default behavior removes the prefix "analise_v2_" and keeps the rest
of the filename intact (e.g., "analise_v2_BTCUSDT_4h.csv" -> "BTCUSDT_4h.csv").

Usage examples:
  python rename_raw_files.py
  python rename_raw_files.py --dry-run
  python rename_raw_files.py --prefix analise_v2_ --directory ./data/enriched
"""

import argparse
import os
from typing import Tuple


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rename files in a directory by stripping a specific prefix from filenames."
    )
    parser.add_argument(
        "--directory",
        "-d",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "enriched"),
        help="Target directory containing files to rename (default: project_root/data/enriched)",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        default="analise_v2_",
        help="Filename prefix to remove (default: analise_v2_)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without performing changes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination if it already exists (otherwise skip).",
    )
    return parser


def compute_destination_name(source_name: str, prefix_to_remove: str) -> Tuple[bool, str]:
    """Return a tuple (should_rename, destination_name)."""
    if not source_name.startswith(prefix_to_remove):
        return False, source_name
    return True, source_name[len(prefix_to_remove) :]


def rename_files_in_directory(
    target_directory: str, prefix_to_remove: str, dry_run: bool, force_overwrite: bool
) -> None:
    if not os.path.isdir(target_directory):
        raise FileNotFoundError(f"Directory not found: {target_directory}")

    files_processed = 0
    files_renamed = 0
    files_skipped = 0
    files_conflicts = 0

    for entry_name in os.listdir(target_directory):
        source_path = os.path.join(target_directory, entry_name)

        # Only process regular files
        if not os.path.isfile(source_path):
            continue

        files_processed += 1
        should_rename, destination_name = compute_destination_name(entry_name, prefix_to_remove)
        if not should_rename:
            files_skipped += 1
            continue

        destination_path = os.path.join(target_directory, destination_name)

        if os.path.exists(destination_path) and not force_overwrite:
            files_conflicts += 1
            print(
                f"SKIP (exists): '{source_path}' -> '{destination_path}' (use --force to overwrite)"
            )
            continue

        print(f"RENAME: '{source_path}' -> '{destination_path}'")
        if not dry_run:
            # If force_overwrite, os.replace will overwrite atomically if supported
            os.replace(source_path, destination_path)
        files_renamed += 1

    print("\n--- Rename Summary ---")
    print(f"Directory: {target_directory}")
    print(f"Prefix removed: '{prefix_to_remove}'")
    print(f"Files scanned: {files_processed}")
    print(f"Files renamed: {files_renamed}")
    print(f"Files skipped (no prefix): {files_skipped}")
    print(f"Files conflicted (existing dest): {files_conflicts}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    rename_files_in_directory(
        target_directory=os.path.abspath(args.directory),
        prefix_to_remove=args.prefix,
        dry_run=args.dry_run,
        force_overwrite=args.force,
    )


if __name__ == "__main__":
    main()


