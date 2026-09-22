"""Hardware / runtime audit for Morves Local STT.

Collects machine facts safely (read-only, no driver or global changes):
OS, CPU, RAM, disk, Python, GPU vendor/model/VRAM/driver, CUDA availability
via ctranslate2 (faster-whisper runtime) and torch (if installed), ffmpeg
presence. Prints JSON; optionally writes it to a file.

Usage:
  python scripts/inspect_hardware.py [--out evaluation/reports/hardware.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _windows_cpu() -> dict[str, object]:
    out = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "$c=Get-CimInstance Win32_Processor; "
            "Write-Output $c.Name; Write-Output $c.NumberOfCores; "
            "Write-Output $c.NumberOfLogicalProcessors; "
            "Write-Output ([math]::Round((Get-CimInstance Win32_ComputerSystem)"
            ".TotalPhysicalMemory/1GB,2)); "
            "Write-Output ([math]::Round((Get-CimInstance Win32_OperatingSystem)"
            ".FreePhysicalMemory/1MB,2))",
        ]
    )
    if not out:
        return {}
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) < 5:
        return {}
    return {
        "cpu": lines[0],
        "physical_cores": int(lines[1]),
        "logical_cores": int(lines[2]),
        "total_ram_gb": float(lines[3]),
        "free_ram_gb": float(lines[4]),
    }


def _nvidia() -> dict[str, object]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if not out:
        return {"present": False}
    row = next((ln for ln in out.splitlines() if ln.strip()), "")
    parts = [p.strip() for p in row.split(",")]
    if len(parts) < 4:
        return {"present": True, "raw": row}
    return {
        "present": True,
        "vendor": "NVIDIA",
        "model": parts[0],
        "driver_version": parts[1],
        "vram_mb": int(parts[2].split()[0]),
        "compute_capability": parts[3],
    }


def _ctranslate2() -> dict[str, object]:
    try:
        import ctranslate2  # type: ignore[import-not-found]

        return {
            "installed": True,
            "version": ctranslate2.__version__,
            "cuda_device_count": ctranslate2.get_cuda_device_count(),
        }
    except Exception as exc:
        return {"installed": False, "reason": str(exc)}


def _torch() -> dict[str, object]:
    try:
        import torch  # type: ignore[import-not-found]

        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
        }
    except Exception:
        return {
            "installed": False,
            "reason": "torch not installed (not required for STT-0 inference)",
        }


def collect() -> dict[str, object]:
    info: dict[str, object] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "disk_free_gb": round(shutil.disk_usage(Path.cwd()).free / (1024**3), 1),
        "ffmpeg_on_path": shutil.which("ffmpeg") is not None,
        "gpu": _nvidia(),
        "ctranslate2": _ctranslate2(),
        "torch": _torch(),
    }
    if platform.system() == "Windows":
        info.update(_windows_cpu())
    else:
        info["cpu"] = platform.processor() or "unknown"
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None, help="Optional path to write JSON")
    args = parser.parse_args()

    info = collect()
    text = json.dumps(info, indent=2, ensure_ascii=False)
    print(text)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"\nwritten: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
