# mlx-restoreformer-swift

MLX-Swift port of **[RestoreFormer++](https://github.com/wzhouxiff/RestoreFormerPlusPlus)**
(TPAMI 2023) — blind face restoration through a VQ-GAN over a reconstruction-oriented
high-quality dictionary (ROHQD) with multi-scale multi-head cross-attention — wrapped as an
MLXEngine `imageRestore` package with the Apple-Vision detect → align → restore →
paste-back pipeline shared with `mlx-gfpgan-swift`.

The second **face** restorer on `imageRestore`: GFPGAN hallucinates from a StyleGAN2
generative prior (stronger detail, more identity drift); RestoreFormer++ reconstructs from
a codebook of real HQ facial features with cross-attention to the degraded input (more
conservative, better fidelity on heavy degradation).

```
Sources/RestoreFormerMLXCore  the network: VQVAEGANMultiHeadTransformer (NHWC, isomorphic
                              to RestoreFormer/modules/vqvae/vqvae_arch.py)
Sources/MLXRestoreFormer      the ModelPackage: Vision face detect/align (FFHQ 5-point),
                              per-face restore, feathered paste-back, strength dial
Sources/Gate                  restoreformer-gate — parity gates vs the PyTorch oracle
Sources/Validate              restoreformer-validate — split footprint through the real engine
oracle/                       torch oracle: convert.py · gen_goldens.py (59 goldens) · publish.py
```

## Weights

`mlx-community/RestoreFormerPlusPlus-fp32` (293.9 MB fp32, 441 tensors, MLX NHWC).
Source: the author's official `RestoreFormer++.ckpt` (v1.0.0 GitHub release), `vqvae.*`
state re-exported through the instantiated arch. Plain **Apache-2.0** — no third-party
carve-outs (contrast GFPGAN's NVIDIA/DFDNet clauses).

**fp32 deliberately.** Measured dtype gates (face fixture): fp16 50.1 dB (viable —
mantissa-bound, it *beats* bf16), bf16 38.6 dB. Late-decoder activations reach ±14k —
close enough to fp16's ceiling that fp32 stays the ship per the restoration-family
precedent.

## Gates (all green, 2026-07-27, M5 Max)

| Gate | Result |
|---|---|
| S0 key contract | 441/441 tensors, 73,472,579 params, strict both ways |
| S1 components | 7/7 (ResnetBlock ±nin, self/cross-attn, resample, VQ) rel ≤ 3e-6, indices 256/256 exact |
| S3 full model | 40/40 per-stage taps on 512² rand + real face; codebook indices 256/256 exact on both |
| Conformance | 12/12 offline (MAT, CAN cadence=face, manifest, strength, align math) |
| Validate (real engine) | floor 0.31 GB · peak 1.79 GB · act 1.47 GB · 1.8 s @1297×1920 ×2 faces |

Fully deterministic — no noise injection anywhere; the codebook argmin is gated on **exact
index equality**, not just tensor tolerance. Per-level `eval` + cache seams in BOTH encoder
and decoder are load-bearing (without the encoder's, activation measured 2.98 GB).

The Vision alignment seam is verbatim from `mlx-gfpgan-swift`, where its agreement gate vs
facexlib ground truth lives (7/7 faces, crop IoU 0.86–0.97 — same template, same crop size).

## Run the gates

```bash
swift run restoreformer-gate --s0 oracle/converted/RestoreFormer++/model.safetensors
swift run restoreformer-gate --all oracle/goldens oracle/converted/RestoreFormer++/model.safetensors
swift run restoreformer-gate --fp16 oracle/goldens oracle/converted/RestoreFormer++/model.safetensors
swift run restoreformer-validate oracle/converted/RestoreFormer++/model.safetensors photo.png
```

## License

Apache-2.0 (port code and weights; upstream is plain Apache-2.0). Residual: ROHQD/model
trained on FFHQ (CC-BY-NC-SA dataset compilation — the standard unsettled question).
