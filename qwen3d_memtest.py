"""Qwen-3D load test: does the 3B model + checkpoint fit on this GPU, and how much room is left?

Run from the qwen3d repo root, in the qwen3d conda env:
    python qwen3d_memtest.py
"""
import os
import time

os.environ.setdefault("WANDB_MODE", "disabled")  # never block on the wandb login prompt

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser

from train import Trainer, setup

GB = 1024 ** 3

args = default_argument_parser().parse_args([
    "--config-file", "qwen3d/configs/qwen_3d.yaml",
    "--eval-only",
    "QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct",
    "MODEL.WEIGHTS", "ckpts/qwen3d_3b.pth",
    "OUTPUT_DIR", "/tmp/qwen3d_memtest",
    "USE_WANDB", "False",
])
cfg = setup(args)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model = Trainer.build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()
torch.cuda.synchronize()

total = torch.cuda.get_device_properties(0).total_memory
alloc = torch.cuda.memory_allocated()
n_params = sum(p.numel() for p in model.parameters())
dtypes = {}
for p in model.parameters():
    dtypes[p.dtype] = dtypes.get(p.dtype, 0) + p.numel()

print(f"\nloaded in {time.time() - t0:.1f} s")
print(f"parameters: {n_params / 1e9:.2f} B  by dtype: "
      + ", ".join(f"{d}: {n / 1e9:.2f} B" for d, n in dtypes.items()))
print(f"GPU memory after load: {alloc / GB:.2f} GB of {total / GB:.2f} GB "
      f"(peak {torch.cuda.max_memory_allocated() / GB:.2f} GB)")
print(f"headroom for activations: {(total - alloc) / GB:.2f} GB")
