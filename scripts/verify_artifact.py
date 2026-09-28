from pathlib import Path
import hashlib, json, zipfile, sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_METHOD = "f02ccffec4bd20225e261bd4a2b88c803ebfbf700e8f53d6b664e42f82dc307d"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)

for rel in ["frozen_cpu/v27_psc_method.py", "frozen_gpu/v27_psc_method.py"]:
    got = sha256(ROOT / rel)
    if got != EXPECTED_METHOD:
        fail(f"{rel} hash {got} != frozen {EXPECTED_METHOD}")
print("PSC-BS frozen source hash: PASS")

contexts = list((ROOT / "frozen_gpu/frozen_contexts").glob("*.json"))
if len(contexts) != 180:
    fail(f"expected 180 frozen GPU contexts, got {len(contexts)}")
print("Frozen GPU contexts:", len(contexts))

cpu_zip = ROOT / "results/V27_FRESH_CPU_RESULTS.zip"
with zipfile.ZipFile(cpu_zip) as z:
    name = "runs/v2_7_fresh_cpu/analysis/SUMMARY.json"
    if name not in z.namelist(): fail("CPU SUMMARY.json missing")
    s = json.loads(z.read(name))
    if s.get("tasks_total") not in (5760, None) and s.get("task_files") not in (5760, None):
        fail(f"unexpected CPU task count fields: {s}")
    if s.get("error", s.get("errors", 0)) not in (0, None): fail("CPU errors nonzero")
print("Fresh CPU summary: PASS")

gpu_zip = ROOT / "results/V27_FRESH_GPU_RESULTS.zip"
with zipfile.ZipFile(gpu_zip) as z:
    name = "runs/v2_7_gpu_fresh/analysis/SUMMARY.json"
    if name not in z.namelist(): fail("GPU SUMMARY.json missing")
    s = json.loads(z.read(name))
    if s.get("task_files") != 2880 or s.get("ok") != 2880 or s.get("error") != 0:
        fail(f"unexpected GPU task summary: {s}")
    if s.get("paired_cells") != 360 or not s.get("all_9_gates_pass"):
        fail(f"unexpected GPU gate summary: {s}")
print("Fresh GPU summary: PASS")

print("ARTIFACT VERIFICATION PASS")
