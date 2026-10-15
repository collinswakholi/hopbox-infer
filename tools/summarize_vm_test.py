"""Compare saved production-EXE CSVs after the offline Windows VM finishes."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');parser.add_argument('--area',type=Path);args=parser.parse_args()
version=json.loads((ROOT/'app/hopbox/branding.json').read_text())['version']
AREA = args.area.resolve() if args.area else ROOT/'validation'/f'vm-{version}'
if args.gpu: AREA = AREA / 'gpu'
vm = json.loads((AREA / "results/vm-result.json").read_text(encoding="utf-8-sig"))
if not vm["pass"] or vm["phase"] != "complete":
    raise RuntimeError("The EXE VM test has not passed.")
prepared = json.loads((AREA / "prepared.json").read_text())
installer = Path(prepared.get('installer_path', ROOT / 'release' / f'HopBox-Infer-Setup-{version}.exe'))
with installer.open("rb") as stream:
    installer_hash = hashlib.file_digest(stream, "sha256").hexdigest()
if installer_hash != prepared["installer_sha256"]:
    raise RuntimeError("The installer has changed since VM preparation.")


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def compare(reference, actual):
    columns, before = table(reference)
    actual_columns, after = table(actual)
    if columns != actual_columns or len(before) != len(after):
        raise RuntimeError("CSV structure changed.")
    fields = []
    identifiers = ("Image_name", "ID", "QR_info", "CC_model")
    for index, (a, b) in enumerate(zip(before, after)):
        for name in identifiers:
            if a[name] != b[name]:
                raise RuntimeError(f"Identity changed in row {index}: {name}")
    for column in columns:
        if column in ("Image_name", "QR_info", "CC_model"):
            continue
        field = {"column": column, "exact": True, "max_absolute_difference": 0.0,
                 "max_relative_difference": 0.0, "absolute_worst": None, "relative_worst": None}
        for a, b in zip(before, after):
            if not a[column] or not b[column]:
                if a[column] != b[column]:
                    raise RuntimeError(f"Missing values changed: {column}")
                continue
            x, y = float(a[column]), float(b[column])
            if not math.isfinite(x) or not math.isfinite(y):
                if x == y or math.isnan(x) and math.isnan(y):
                    continue
                raise RuntimeError(f"Nonfinite values changed: {column}")
            absolute = abs(x - y)
            relative = absolute / max(abs(x), 1e-12)
            worst = {"image": a["Image_name"], "id": a["ID"], "before": x, "after": y}
            if absolute:
                field["exact"] = False
            if absolute > field["max_absolute_difference"]:
                field.update(max_absolute_difference=absolute, absolute_worst=worst)
            if relative > field["max_relative_difference"]:
                field.update(max_relative_difference=relative, relative_worst=worst)
        fields.append(field)
    area = next(f for f in fields if f["column"] == "RP_Area")
    if area["max_relative_difference"] > 0.001:
        raise RuntimeError("Area parity gate failed.")
    return {"rows": len(after), "images": len({r["Image_name"] for r in after}),
            "identifiers_match": True, "area_gate_pass": True,
            "all_numeric_fields_exact": all(f["exact"] for f in fields), "fields": fields}


baseline = AREA / "results/batch-1/119CANON/Features.csv"
records = []
for batch in sorted(vm["batches"], key=lambda b: b["batch"]):
    actual = AREA / f"results/batch-{batch['batch']}/119CANON/Features.csv"
    records.append({"batch": batch["batch"], "seconds": batch["seconds"],
                    "versus_batch_1": compare(baseline, actual),
                    "versus_nvidia_host": compare(AREA / "control/reference.csv", actual)})
record = {"pass": True, "source": "installed HopBox-Infer.exe in an offline Windows Sandbox",
          "mode": "DirectML GPU" if args.gpu else "CPU",
          "installer_sha256": installer_hash,
          "batches": records,
          "all_batches_exact": all(r["versus_batch_1"]["all_numeric_fields_exact"] for r in records),
          "all_vs_nvidia_fields_exact": all(r["versus_nvidia_host"]["all_numeric_fields_exact"] for r in records)}
if vm.get('cpu_smoke', {}).get('pass'):
    record['cpu_smoke_vs_nvidia_host'] = compare(
        AREA/'control/reference.csv', AREA/'results/cpu-smoke/119CANON/Features.csv')
(AREA / "comparison.json").write_text(json.dumps(record, indent=2))
prepared.update(executed=True, passed=True, guest_started_utc=vm["started_utc"],
                guest_completed_utc=vm["updated_utc"])
(AREA / "prepared.json").write_text(json.dumps(prepared, indent=2))
print("VM EXE CSV comparisons passed:", len(records), "batches;",
      "all batches exact:", record["all_batches_exact"],
      "; versus NVIDIA exact:", record["all_vs_nvidia_fields_exact"])
