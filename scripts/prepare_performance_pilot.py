"""Create independent source snapshots without touching the benchmark checkout."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="63454a0")
    parser.add_argument("--output", type=Path, default=Path("results/performance"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=repo)

    commit = git("rev-parse", args.baseline_ref + "^{commit}").decode().strip()
    roots = [output / "frozen", output / "candidate"]
    if any(path.exists() for path in roots):
        raise SystemExit("Snapshot already exists; choose a new output directory. Existing experiments are never overwritten.")
    paths = git("ls-tree", "-r", "--name-only", commit).decode().splitlines()
    selected = [p for p in paths if p.startswith(("src/", "scripts/", "tests/")) or
                p in ("requirements.txt", "requirements-lock.txt", "pytest.ini")]
    manifest = {"baseline_commit": commit, "candidate_commit": git("rev-parse", "HEAD").decode().strip(),
                "candidate_working_tree": git("status", "--porcelain").decode(), "snapshots": {}}
    for root in roots:
        hashes = {}
        snapshot_paths = selected if root.name == "frozen" else sorted(set(
            git("ls-files", "--cached", "--others", "--exclude-standard", "src", "scripts", "tests").decode().splitlines()
            + [p for p in selected if "/" not in p]))
        for relative in snapshot_paths:
            data = git("show", commit + ":" + relative) if root.name == "frozen" else (repo / relative).read_bytes()
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            hashes[relative] = hashlib.sha256(data).hexdigest()
        manifest["snapshots"][root.name] = hashes
    (output / "snapshots.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared independent frozen and candidate snapshots in {output}")


if __name__ == "__main__":
    main()
