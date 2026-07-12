"""
List qBraid devices visible to your account (id, qubits, status, queue).

Auth comes from the saved qBraid config (~/.qbraid/qbraidrc) or the
QBRAID_API_KEY environment variable — no secret is stored in this repo.

    python experiments/qbraid_list_devices.py            # all devices
    python experiments/qbraid_list_devices.py --online   # only ONLINE ones
    python experiments/qbraid_list_devices.py --gate      # hide analog QuEra/Pasqal

Use this to find a live gate QPU before running experiments/qbraid_hardware_qrc.py.
Note: QuEra (aquila) and Pasqal (fresnel) are ANALOG neutral-atom devices and do
NOT run the gate-based reservoir circuits — they are flagged [analog] below.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qrc.hardware_backend import qbraid_api_key

# Analog (neutral-atom / AHS) devices that cannot run gate circuits.
_ANALOG = ("quera", "pasqal")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--online", action="store_true", help="only show ONLINE devices")
    p.add_argument("--gate", action="store_true", help="hide analog (QuEra/Pasqal) devices")
    p.add_argument("--pricing", action="store_true",
                   help="show per-task / per-shot pricing reported by qBraid")
    args = p.parse_args()

    from qbraid.runtime import QbraidProvider

    provider = QbraidProvider(api_key=qbraid_api_key())
    rows = []
    for d in provider.get_devices():
        try:
            md = d.metadata()
        except Exception as e:  # pragma: no cover
            md = {"status": f"ERR:{str(e)[:30]}"}
        dev_id = str(getattr(d, "id", md.get("device_id", "?")))
        status = str(md.get("status", "?"))
        qubits = str(md.get("num_qubits", md.get("number_qubits", "?")))
        queue = str(md.get("queue_depth", md.get("pending_jobs", "?")))
        analog = any(v in dev_id for v in _ANALOG)
        kind = "analog" if analog else ("sim" if ":sim:" in dev_id else "gate-qpu")
        if args.online and status.upper() != "ONLINE":
            continue
        if args.gate and analog:
            continue
        pricing = md.get("pricing", {}) or {}
        per_task = pricing.get("perTask", pricing.get("per_task", "?"))
        per_shot = pricing.get("perShot", pricing.get("per_shot", "?"))
        rows.append((dev_id, kind, qubits, status, queue, per_task, per_shot))

    rows.sort(key=lambda r: (r[1], r[0]))
    suffix = "  per_task  per_shot" if args.pricing else ""
    print(f"{'device_id':<42} {'kind':<9} {'qubits':<7} {'status':<11} queue{suffix}")
    print("-" * (105 if args.pricing else 82))
    for r in rows:
        tag = "  [analog: gate circuits NOT supported]" if r[1] == "analog" else ""
        price = f"  {str(r[5]):<8}  {r[6]}" if args.pricing else ""
        print(f"{r[0]:<42} {r[1]:<9} {r[2]:<7} {r[3]:<11} {r[4]}{price}{tag}")
    print(f"\nTotal: {len(rows)} devices")


if __name__ == "__main__":
    main()
