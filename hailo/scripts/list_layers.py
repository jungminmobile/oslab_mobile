import sys, json
from hailo_sdk_client import ClientRunner
r = ClientRunner(har=sys.argv[1])
hn = r.get_hn_dict() if hasattr(r, "get_hn_dict") else json.loads(r.get_hn())
for name, L in hn["layers"].items():
    t = L.get("type")
    if t not in ("conv", "ew_add") or any(k in name for k in sys.argv[2:]):
        q = L.get("quantization_params", {}) or {}
        print(name.split("/")[-1].ljust(24), str(t).ljust(18), ";".join(L.get("original_names", []))[:70], str(L.get("precision_config", ""))[:60])
