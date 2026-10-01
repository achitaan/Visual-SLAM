"""Atomic benchmark JSON output with brief Windows sharing-violation retries."""
import json
import os
from pathlib import Path
import time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(f'.{os.getpid()}.json.part')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False))
    for attempt in range(20):
        try:
            pending.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)
