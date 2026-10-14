"""Outer-owned candidate arm; old frozen source remains the default binding."""
from pathlib import Path
import json
import os
import re
import sys

from native_application_arm_owner_v4 import ENV_INNER, run_owned_arm
from native_application_stop_owner_v2 import STOP_RUN, STOP_DIGEST


HERE = Path(__file__).resolve()
B = HERE.parent


def main() -> None:
    if os.environ.get(ENV_INNER) == "1":
        if len(sys.argv) != 4:
            raise ValueError("owned candidate inner requires protocol, name, cap")
        sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
        from native_application_live_driver_core_v2 import run
        run("candidate_identity", "native-origin-serialized-build.json",
            "compiled_isolated_serialization_pending_pipeline_qualification",
            "native-origin-serialized-source.json", HERE)
        return
    if len(sys.argv) != 5 or not re.fullmatch(r"[a-z0-9-]+\.json", sys.argv[1]) or not re.fullmatch(r"[a-z0-9_]+", sys.argv[2]):
        raise ValueError("candidate outer owner requires PROTOCOL NAME CAP")
    protocol, name, cap_text, control_digest = sys.argv[1:]
    if not re.fullmatch(r"[a-f0-9]{64}", control_digest):
        raise ValueError("application Stop digest required")
    cap = float(cap_text)
    proof = run_owned_arm(
        command=[sys.executable, "-B", str(HERE), protocol, name, cap_text],
        name=name, cap_s=cap, control_root=B, case_path=B / name / "case.json",
        extra_env={STOP_RUN: name, STOP_DIGEST: control_digest},
        cleanup_control_paths=(B / f"{name}-finalizer-request.json",))
    print(json.dumps({"status": proof["status"], "name": name, "cap_s": cap}, sort_keys=True))


if __name__ == "__main__":
    main()
