#!/usr/bin/env python3
"""Standalone GPU interop A/B for comfy-kitchen issue #245.

Run in a *fresh process* using the selected Conda Python after installing the
Kitchen wheel. Use the same command against static and experimental shared
cudart builds. Do not confuse a successful first CUDA synchronization with a
successful second PyTorch GPU launch.
"""
import argparse
import os
import subprocess
import sys
import traceback

import torch
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("quantize", "sol"), default="quantize")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    torch.cuda.init()
    device = torch.device("cuda:0")
    print("Torch:", torch.__version__, "torch CUDA:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(device), flush=True)
    print("Kitchen extension:", cuda._C.__file__, flush=True)
    print("Current stream:", hex(torch.cuda.current_stream(device).cuda_stream), flush=True)
    libraries = sorted({
        line.split()[-1] for line in open("/proc/self/maps", encoding="utf8")
        if "libcudart.so" in line
    })
    print("Mapped shared CUDART:", libraries, flush=True)
    try:
        result = subprocess.run(["readelf", "-d", cuda._C.__file__],
                                capture_output=True, text=True, check=False)
        print("Extension CUDART DT_NEEDED:",
              [line.strip() for line in result.stdout.splitlines() if "libcudart" in line],
              flush=True)
    except FileNotFoundError:
        print("readelf unavailable", flush=True)
    torch.manual_seed(0)
    if args.operation == "quantize":
        x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    else:
        q, k, v = (torch.randn(1, 4096, 2, 128, device=device,
                               dtype=torch.bfloat16) for _ in range(3))
    torch.cuda.synchronize()
    for i in range(args.iterations):
        print(f"Iteration {i+1}: native call", flush=True)
        if args.operation == "quantize":
            output = cuda.quantize_int8_rowwise(x)
        else:
            output = ck.sol_attn(q, k, v, tau=4.0, token_aug=0)
        torch.cuda.synchronize()
        print("Native sync: PASS", flush=True)
        tmp = torch.empty(64, device=device)
        print("PyTorch allocate: PASS", flush=True)
        tmp.fill_(1.0)
        torch.cuda.synchronize()
        print("PyTorch fill/sync: PASS", flush=True)
    print("RESULT: PASS", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("RESULT: FAIL", flush=True)
        sys.exit(1)
