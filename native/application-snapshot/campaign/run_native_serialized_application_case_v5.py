"""Journaled application binding for the frozen serialized native candidate."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

from native_application_admission_v4 import (
    ENV_JOURNAL, ENV_TOKEN, JournaledOwnedPausedProcess)
from native_application_arm_owner_v4 import ENV_INNER
from native_application_arm_owner_v6 import run_owned_arm
from native_application_stop_owner_v2 import STOP_DIGEST, STOP_RUN


HERE = Path(__file__).resolve()
B = HERE.parent


def main() -> None:
    if os.environ.get(ENV_INNER) == "1":
        if len(sys.argv) != 4:
            raise ValueError("journaled candidate inner requires protocol, name, cap")
        sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
        # Bind journaled ownership at the two existing import seams without
        # changing estimator, decoder, finalizer, or numerical source bytes.
        import native_application_live_driver_core_v2 as core
        import native_application_raw_finalizer as finalizer
        finalizer.OwnedPausedProcess = JournaledOwnedPausedProcess
        core.OwnedPausedProcess = JournaledOwnedPausedProcess
        core.finalize_owned = finalizer.finalize_owned
        core.run("candidate_identity", "native-origin-serialized-build.json",
                 "compiled_isolated_serialization_pending_pipeline_qualification",
                 "native-origin-serialized-source.json", HERE)
        return
    if (len(sys.argv) != 6 or not re.fullmatch(r"[a-z0-9-]+\.json", sys.argv[1])
            or not re.fullmatch(r"native_prototype_[a-z0-9_]+", sys.argv[2])):
        raise ValueError("journaled candidate owner requires PROTOCOL NAME CAP DIGEST TOKEN")
    protocol, name, cap_text, control_digest, admission_token = sys.argv[1:]
    if (not re.fullmatch(r"[a-f0-9]{64}", control_digest)
            or not re.fullmatch(r"[a-f0-9]{64}", admission_token)):
        raise ValueError("application Stop digest and admission token required")
    cap = float(cap_text)
    journal = B / f"{name}-admissions-v3.jsonl"
    proof = run_owned_arm(
        command=[sys.executable, "-B", str(HERE), protocol, name, cap_text],
        name=name, cap_s=cap, control_root=B, case_path=B / name / "case.json",
        control_digest=control_digest, admission_token=admission_token,
        journal_path=journal,
        extra_env={STOP_RUN: name, STOP_DIGEST: control_digest,
                   ENV_JOURNAL: str(journal), ENV_TOKEN: admission_token},
        cleanup_control_paths=(B / f"{name}-finalizer-request.json",))
    print(json.dumps({"status": proof["status"], "name": name, "cap_s": cap},
                     sort_keys=True))


if __name__ == "__main__":
    main()
