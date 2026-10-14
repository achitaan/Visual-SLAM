"""Resource forecast successor; every numerical preset remains pinned by v1."""
import hashlib
import json
from pathlib import Path
from native_application_presets_v1 import select_preset as original_select
B=Path(__file__).resolve().parent
DESIGN=B/"native-bounded-full01-measured-resource-design.json"
EXPECTED="c2301f715ced1abf37b803e1b76e555379fa2ee4dc48088fc15c2ee07846d12a"
def select_preset(sequence,frames,seed):
    value=original_select(sequence,frames,seed)
    if hashlib.sha256(DESIGN.read_bytes()).hexdigest()!=EXPECTED:raise ValueError("resource design changed")
    design=json.loads(DESIGN.read_text(encoding="utf8"))
    for name,digest in design["evidence_sha256"].items():
        if hashlib.sha256((B/name).read_bytes()).hexdigest()!=digest:raise ValueError("resource evidence changed")
    if (sequence,frames,seed)==("01",1101,0):
        assert value["output_forecast_bytes"]==design["previous_forecast_bytes"]
        value=dict(value,output_forecast_bytes=design["new_output_forecast_bytes"],
            previous_output_forecast_bytes=value["output_forecast_bytes"],
            measured_resource_design_sha256=EXPECTED,resource_forecast_scope="new bounded application recording only")
    return value
