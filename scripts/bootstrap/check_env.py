#!/usr/bin/env python3
"""Emit a machine-readable snapshot of the remote ESS environment."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - defensive probe path
        return {
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def probe_gpu_host(nvidia_smi_query: dict[str, object] | None) -> bool:
    return bool(
        nvidia_smi_query
        and nvidia_smi_query.get("returncode") == 0
        and nvidia_smi_query.get("stdout")
    )


def probe_torch(gpu_host_present: bool) -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "status": "FAIL",
        }

    cuda_available = bool(torch.cuda.is_available())
    payload = {
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "status": "PASS",
    }
    if gpu_host_present and not cuda_available:
        payload["status"] = "FAIL"
        payload["warning"] = (
            "GPU host detected but torch.cuda.is_available() is false."
        )
    return {
        **payload,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    nvidia_smi_path = shutil.which("nvidia-smi")
    nvidia_smi_query = None
    if nvidia_smi_path:
        nvidia_smi_query = run_command(
            [
                nvidia_smi_path,
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
    gpu_host_present = probe_gpu_host(nvidia_smi_query)
    torch_probe = probe_torch(gpu_host_present)

    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "project_root": str(project_root),
        "project_root_exists": project_root.exists(),
        "gpu_host_present": gpu_host_present,
        "nvidia_smi": {
            "path": nvidia_smi_path,
            "query": nvidia_smi_query,
        },
        "torch": torch_probe,
    }

    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")

    if gpu_host_present and not torch_probe.get("cuda_available"):
        sys.stderr.write(
            "ERROR: GPU host detected but torch.cuda.is_available() is false.\n"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
