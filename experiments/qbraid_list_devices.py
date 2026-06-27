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

# Analog (neutral-atom / AHS) devices that cannot run gate circuits.
_ANALOG = ("quera", "pasqal")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--online", action="store_true", help="only show ONLINE devices")
    p.add_argument("--gate", action="store_true", help="hide analog (QuEra/Pasqal) devices")
    args = p.parse_args()

    from qbraid.runtime import QbraidProvider

    provider = QbraidProvider()
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
        rows.append((dev_id, kind, qubits, status, queue))

    rows.sort(key=lambda r: (r[1], r[0]))
    print(f"{'device_id':<42} {'kind':<9} {'qubits':<7} {'status':<11} queue")
    print("-" * 82)
    for r in rows:
        tag = "  [analog: gate circuits NOT supported]" if r[1] == "analog" else ""
        print(f"{r[0]:<42} {r[1]:<9} {r[2]:<7} {r[3]:<11} {r[4]}{tag}")
    print(f"\nTotal: {len(rows)} devices")


if __name__ == "__main__":
    main()
