"""Phase 1 - environment check. Run: python scripts/env_check.py
Writes results/reports/env_report.json and prints a summary."""
import json, os, platform, subprocess, sys, importlib
from pathlib import Path

report = {"python": sys.version, "executable": sys.executable,
          "platform": platform.platform(), "machine": platform.machine()}

def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

report["nvidia_smi"] = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap",
                            "--format=csv,noheader"])

try:
    import psutil
    vm = psutil.virtual_memory()
    report["ram_total_gb"] = round(vm.total / 1e9, 1)
    report["ram_available_gb"] = round(vm.available / 1e9, 1)
    report["cpu_logical_cores"] = psutil.cpu_count()
except ImportError:
    report["ram"] = "psutil not installed"

libs = {}
for name in ["torch", "numpy", "pandas", "sklearn", "seqeval", "matplotlib",
             "seaborn", "tqdm", "huggingface_hub", "datasets", "pytorch_crf", "torchcrf"]:
    try:
        m = importlib.import_module(name)
        libs[name] = getattr(m, "__version__", "installed")
    except Exception:
        libs[name] = None
report["libraries"] = libs

try:
    import torch
    t = {"version": torch.__version__, "built_cuda": torch.version.cuda,
         "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
         "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        t.update(device=p.name, capability=f"{p.major}.{p.minor}",
                 vram_gb=round(p.total_memory / 1e9, 2),
                 arch_list=torch.cuda.get_arch_list())
        # Real smoke test: an actual cuDNN LSTM forward/backward on the GPU.
        import time
        lstm = torch.nn.LSTM(128, 256, batch_first=True, bidirectional=True).cuda()
        x = torch.randn(64, 40, 128, device="cuda", requires_grad=True)
        torch.cuda.synchronize(); s = time.time()
        for _ in range(20):
            out, _ = lstm(x); out.sum().backward()
        torch.cuda.synchronize()
        t["bilstm_smoke_test"] = f"OK, 20 fwd+bwd steps in {time.time()-s:.2f}s"
        t["bf16_supported"] = torch.cuda.is_bf16_supported()
    report["torch"] = t
except Exception as e:
    report["torch"] = f"FAILED: {type(e).__name__}: {e}"

out = Path("results/reports"); out.mkdir(parents=True, exist_ok=True)
(out / "env_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
print("\nSaved -> results/reports/env_report.json")
