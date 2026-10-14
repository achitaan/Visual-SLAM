"""Bounded same-run Stop client for the existing loopback application control."""
from __future__ import annotations
import argparse
import json
import re
import time

def validate_hello(value: dict, run_id: str) -> str:
    if (not isinstance(value, dict) or value.get("type") != "control_hello"
            or value.get("run_id") != run_id
            or not isinstance(value.get("capability"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", value["capability"])):
        raise ValueError("Control hello identity differs; no Stop sent")
    return value["capability"]

def request_stop(run_id: str) -> dict:
    if not re.fullmatch(r"native_prototype_[a-z0-9_]{1,80}", run_id):
        raise ValueError("Scoped existing application run ID required")
    from websockets.sync.client import connect
    deadline = time.monotonic() + 15
    last_connection_error = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Application Stop control readiness deadline; cleanup unconfirmed; last connection error=" + repr(last_connection_error))
        try:
            websocket = connect("ws://127.0.0.1:8780/", origin="http://127.0.0.1:4301",
                open_timeout=min(2, remaining), close_timeout=.25, proxy=None,
                compression=None, max_size=4096, max_queue=4)
        except OSError as error:
            last_connection_error = (type(error).__name__, str(error))
            # A CtrlC during launcher startup may precede its socket binding.
            # Retry only connection readiness under this one absolute deadline.
            time.sleep(min(.05, max(0, deadline-time.monotonic())))
            continue
        with websocket:
            hello = json.loads(websocket.recv(timeout=max(.001, deadline-time.monotonic())))
            capability = validate_hello(hello, run_id)
            websocket.send(json.dumps(dict(action="stop_run", run_id=run_id, capability=capability)))
            while True:
                response = json.loads(websocket.recv(timeout=max(.001, deadline-time.monotonic())))
                if (not isinstance(response, dict) or response.get("run_id") != run_id
                        or response.get("type") != "control_state"):
                    raise ValueError("Stop response identity differs; cleanup unconfirmed")
                if response.get("cleanup_confirmed") is True and response.get("phase") in ("cancelled", "completed"):
                    # Deliberately omit the capability and other peer fields from stdout.
                    return dict(run_id=run_id, phase=response["phase"], cleanup_confirmed=True)
                if response.get("phase") == "failed" or response.get("status") == "rejected_control":
                    raise ValueError("Application rejected Stop or failed; cleanup unconfirmed")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Stop acknowledgement deadline; cleanup unconfirmed")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(request_stop(args.run_id), allow_nan=False))

if __name__ == "__main__":
    main()
