"""
I/O utilities for handling benchmark results and metadata.

This module provides functions for saving benchmark results to CSV files and metadata to JSONL files.
"""

import json
import os


def _append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _append_df_csv(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False, header=not file_exists)


def save_benchmark(
    results_root,
    subdir,
    run_id,
    suffix,
    df,
    meta,
    existing_results_path=None,
    print_metadata=True,
):
    path = os.path.join(results_root, subdir)
    os.makedirs(path, exist_ok=True)

    if existing_results_path is not None and os.path.exists(existing_results_path):
        csv_file = existing_results_path
    else:
        csv_file = os.path.join(path, f"{run_id}_{suffix}.csv")

    _append_df_csv(csv_file, df)
    if print_metadata:
        _append_jsonl(
            os.path.join(path, f"{run_id}_{suffix}_metadata.jsonl"),
            meta,
        )
