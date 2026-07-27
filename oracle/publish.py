"""Publish the converted RestoreFormer++ weights to mlx-community.

  RestoreFormer++ -> mlx-community/RestoreFormerPlusPlus-fp32
  ("++" is not a legal HF repo id, hence the upstream repo's own PlusPlus spelling)

Run:  .venv/bin/python publish.py [--dry-run]
"""
import os
import sys

from huggingface_hub import HfApi

API = HfApi()
HERE = os.path.dirname(os.path.abspath(__file__))

REPO = "mlx-community/RestoreFormerPlusPlus-fp32"

CARD = """---
license: apache-2.0
library_name: mlx
tags:
- mlx
- face-restoration
- image-restoration
- vqgan
- restoreformer
base_model: wzhouxiff/RestoreFormerPlusPlus
---

# RestoreFormerPlusPlus-fp32 (MLX)

[RestoreFormer++](https://github.com/wzhouxiff/RestoreFormerPlusPlus) (TPAMI 2023) blind
face restoration converted to MLX NHWC safetensors for Apple Silicon. 73,472,579
parameters (441 tensors), fp32.

- **Architecture:** `VQVAEGANMultiHeadTransformer` — VQ-GAN encoder/decoder over a
  1024×256 ROHQD codebook, multi-scale multi-head cross-attention (enc attn @16, dec
  @[16, 32]). 512×512 aligned face crops, RGB in [-1, 1]. Fully deterministic.
- **Source:** the author's official `RestoreFormer++.ckpt` (v1.0.0 GitHub release);
  `vqvae.*` state re-exported through the instantiated architecture.
- **Layout:** MLX NHWC. Conv `(O,kH,kW,I)`; GroupNorm vectors, biases, and the codebook
  embedding pass through. Keys mirror the upstream state dict (prefix stripped).
- **dtype:** fp32. Measured alternatives: fp16 50.1 dB vs the fp32 golden (viable), bf16
  38.6 dB (mantissa-bound — fp16 beats bf16 here). Late-decoder activations reach ±14k.

## License

Apache-2.0 — upstream is plain Apache-2.0 with no third-party carve-outs. Trained on FFHQ
(dataset compilation CC-BY-NC-SA — the standard unsettled dataset-to-weights question).

## Consume

Swift (Apple Silicon): `mlx-restoreformer-swift` —
`RestoreFormerMLXCore.RestoreFormer` + `MLXRestoreFormer.RestoreFormerRestorePackage`
(MLXEngine `imageRestore`, Vision-based detect/align/paste).

Parity vs the PyTorch reference: key contract 441/441 tensors; per-stage taps 40/40
≤ 5e-4 relative (fp32, CPU stream); codebook indices 256/256 exact.
"""


def main():
    dry = "--dry-run" in sys.argv
    src = os.path.join(HERE, "converted", "RestoreFormer++", "model.safetensors")
    assert os.path.exists(src), src
    print(f"{'[dry-run] ' if dry else ''}{src} -> {REPO}")
    if dry:
        print(CARD)
        return
    API.create_repo(REPO, repo_type="model", exist_ok=True)
    API.upload_file(path_or_fileobj=src, path_in_repo="model.safetensors",
                    repo_id=REPO, repo_type="model")
    API.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="model")
    print("✅ published")


main()
