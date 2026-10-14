"""Check the published saved-evidence package without executing SLAM."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Evidence path escapes package: {name}")
        raw = path.read_bytes()
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"Evidence identity mismatch: {name}")
    data = json.loads((root / "results.json").read_text(encoding="utf-8"))
    rows = data["individual_results"]
    if len(rows) != 19 or sum(row["strict_gate_pass"] for row in rows) != 17:
        raise ValueError("Expected 19 comparisons and 17 original strict passes")
    with (root / "results.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(rows):
        raise ValueError("CSV/JSON comparison count differs")
    for csv_row, row in zip(csv_rows, rows):
        for key, value in row.items():
            if csv_row[key] != ("" if value is None else str(value)):
                raise ValueError(f"CSV/JSON value differs: {row['sequence']} {key}")
    for group in data["grouped_results"]:
        selected = [r for r in rows if (r["sequence"], r["frames"], r["scope"]) ==
                    (group["sequence"], group["frames"], group["scope"])]
        if sorted(r["seed"] for r in selected) != group["seeds"]:
            raise ValueError("Grouped seed coverage differs")
        for key in ("vo_ate_rmse_m", "native_ate_rmse_m", "vo_whole_wall_s", "native_whole_wall_s"):
            if median(r[key] for r in selected) != group[key]:
                raise ValueError(f"Grouped median differs: {group['sequence']} {key}")
    print(f"Verified {len(manifest['files'])} original files; 19 comparisons, 17 strict passes, 2 accepted tradeoffs; no replay.")


if __name__ == "__main__":
    main()
