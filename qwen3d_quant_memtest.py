"""Qwen-3D quantized load test: merge LoRA, quantize the language model's linear layers, measure memory.

The bf16 model takes 7.1 GB on the 8 GB 4060, which leaves no room for activations.
Here we fold the LoRA weights into the base model and quantize the 36 decoder
layers to 8-bit (or 4-bit). The vision tower, embeddings and mask decoder stay in bf16.

Run from the qwen3d repo root, in the qwen3d conda env:
    pip install bitsandbytes
    python qwen3d_quant_memtest.py            # int8 (near-lossless)
    python qwen3d_quant_memtest.py nf4        # 4-bit fallback if int8 is still too tight
"""
import os
import sys
import time

os.environ.setdefault("WANDB_MODE", "disabled")

import bitsandbytes as bnb
import torch
import torch.nn as nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser

from train import Trainer, setup

GB = 1024 ** 3
MODE = sys.argv[1] if len(sys.argv) > 1 else "int8"
assert MODE in ("int8", "nf4"), MODE


def gpu(label):
    torch.cuda.synchronize()
    print(f"{label:<28} {torch.cuda.memory_allocated() / GB:5.2f} GB")


def quantize_linear(lin):
    """Return a bitsandbytes replacement for one nn.Linear, quantized on the GPU."""
    w = lin.weight.data.to("cpu")
    b = lin.bias.data.to("cpu") if lin.bias is not None else None
    if MODE == "int8":
        q = bnb.nn.Linear8bitLt(lin.in_features, lin.out_features, bias=b is not None,
                                has_fp16_weights=False, threshold=6.0)
        q.weight = bnb.nn.Int8Params(w.half(), requires_grad=False, has_fp16_weights=False)
    else:
        q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=b is not None,
                              compute_dtype=torch.bfloat16, quant_type="nf4")
        q.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4")
    if b is not None:
        q.bias = nn.Parameter(b, requires_grad=False)
    return q.cuda()  # quantization happens here


args = default_argument_parser().parse_args([
    "--config-file", "qwen3d/configs/qwen_3d.yaml",
    "--eval-only",
    "QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct",
    "MODEL.WEIGHTS", "ckpts/qwen3d_3b.pth",
    "OUTPUT_DIR", "/tmp/qwen3d_memtest",
    "USE_WANDB", "False",
])
cfg = setup(args)

t0 = time.time()
model = Trainer.build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()
gpu("bf16 + LoRA loaded:")

# 1. Fold LoRA into the base weights. model.py checks cfg.USE_LORA to find
#    embed_tokens, so turn it off once the PEFT wrapper is gone.
model.qwen_model = model.qwen_model.merge_and_unload()
cfg.defrost()
cfg.USE_LORA = False
cfg.freeze()
model.cfg = cfg
torch.cuda.empty_cache()
gpu("LoRA merged:")

# 2. Quantize every linear layer in the language decoder, one at a time so the
#    bf16 copy is freed before the next one is built.
layers = model.qwen_model.model.layers
n_swapped = 0
for layer in layers:
    for parent_name in ("self_attn", "mlp"):
        parent = getattr(layer, parent_name)
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, quantize_linear(child))
                del child
                n_swapped += 1
    torch.cuda.empty_cache()
print(f"quantized {n_swapped} linear layers to {MODE} in {time.time() - t0:.1f} s total")

total = torch.cuda.get_device_properties(0).total_memory
alloc = torch.cuda.memory_allocated()
gpu(f"{MODE} model:")
print(f"headroom for activations:    {(total - alloc) / GB:5.2f} GB of {total / GB:.2f} GB")

# 3. Smoke test: one forward through the quantized language model with ~1,500
#    tokens (about the size of a 3-frame scene plus prompt) to measure activations.
torch.cuda.reset_peak_memory_stats()
embeds = torch.randn(1, 1500, model.qwen_model.config.hidden_size,
                     device="cuda", dtype=torch.bfloat16)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    t1 = time.time()
    # The modified Qwen forward calls attention_mask.dim() without a None check.
    mask = torch.ones(embeds.shape[:2], device="cuda", dtype=torch.long)
    out = model.qwen_model.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
    torch.cuda.synchronize()
print(f"1,500-token forward:         {time.time() - t1:5.2f} s, "
      f"peak {torch.cuda.max_memory_allocated() / GB:.2f} GB, "
      f"output finite: {torch.isfinite(out.last_hidden_state).all().item()}")
