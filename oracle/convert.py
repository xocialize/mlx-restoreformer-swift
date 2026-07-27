"""RestoreFormer++ weight conversion: RestoreFormer++.ckpt -> safetensors in MLX NHWC layout.

The release checkpoint is a pytorch-lightning bundle whose `state_dict` carries the model
under a `vqvae.` prefix ALONGSIDE training-only modules (LPIPS, discriminator, ...).
Upstream's own loader strips the prefix and loads `strict=False`. We do it properly:
instantiate the arch, load exactly as upstream does, then re-export the MODEL's own
state_dict — the converted key set is the architecture's, with the loss baggage dropped
and nothing silently missing (any absent model key would survive as random init and fail
parity, so gen_goldens.py also asserts the load hit every key).

Layout: every 4-D conv `(O,I,kH,kW) -> (O,kH,kW,I)`; GroupNorm vectors, the 1024x256
codebook embedding, and biases pass through.

Run:  .venv/bin/python convert.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "upstream"))

spec = importlib.util.spec_from_file_location(
    "vqvae_arch", os.path.join(HERE, "upstream", "RestoreFormer", "modules", "vqvae", "vqvae_arch.py"))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

STEM = "RestoreFormer++"
raw = torch.load(os.path.join(HERE, "weights", f"{STEM}.ckpt"),
                 map_location="cpu", weights_only=False)
weights = raw["state_dict"]
stripped = {k.replace("vqvae.", "", 1): v for k, v in weights.items() if k.startswith("vqvae.")}
dropped = [k for k in weights if not k.startswith("vqvae.")]

model = A.VQVAEGANMultiHeadTransformer()   # the ++ defaults ARE the release config
missing, unexpected = model.load_state_dict(stripped, strict=False)
assert not missing, f"model keys absent from checkpoint: {missing}"
print(f"vqvae.* keys: {len(stripped)}   non-model keys dropped: {len(dropped)}   "
      f"unexpected (training-only) vqvae keys ignored: {len(unexpected)}")

sd = model.state_dict()
out_dir = os.path.join(HERE, "converted", STEM)
os.makedirs(out_dir, exist_ok=True)

converted, stats = {}, {"conv4d": 0, "passthrough": 0}
for k, v in sd.items():
    a = v.detach().cpu().numpy().astype(np.float32)
    if a.ndim == 4:
        a = np.transpose(a, (0, 2, 3, 1))
        stats["conv4d"] += 1
    else:
        stats["passthrough"] += 1
    converted[k] = np.ascontiguousarray(a)

total = sum(int(np.prod(v.shape)) for v in converted.values())
print(f"=== {STEM} ===")
print(f"  tensors: {len(converted)}   transforms: conv4d {stats['conv4d']} "
      f"· passthrough {stats['passthrough']}")
print(f"  params: {total:,}  ({total * 4 / 1e6:.2f} MB fp32)")

meta = {"format": "pt", "source": f"wzhouxiff/RestoreFormerPlusPlus {STEM}.ckpt (state_dict, vqvae.*)",
        "license": "Apache-2.0",
        "layout": "MLX NHWC; conv (O,kH,kW,I)", "params": str(total)}
save_file(converted, os.path.join(out_dir, "model.safetensors"), metadata=meta)
with open(os.path.join(out_dir, "CONVERSION.json"), "w") as f:
    json.dump({"stem": STEM, "transforms": stats, "params": total,
               "dropped_non_model": len(dropped)}, f, indent=2)
print(f"  written: {out_dir}/model.safetensors "
      f"({os.path.getsize(os.path.join(out_dir, 'model.safetensors')) / 1e6:.2f} MB)")
