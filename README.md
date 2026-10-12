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

**Engine-materialized weights read in place since 0.2.1 (AB-A-0112).** MLXEngine downloads the declared
`weightSources` into its model store before `load()`; `load()` reads that copy (flat or snapshot layout) instead of
fetching a second one through swift-transformers' `HubApi`. It also reads `<root>/models/<org>/<name>/`, where earlier
versions downloaded, so existing installs aren't fetched again. `HubApi` remains only the backstop outside the engine
(no store, or nothing materialized); an explicit `weightsURL` bypasses all of it.

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

## GPU numerics: mlx's lossy Winograd conv2d window (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses about
6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32.

RestoreFormer++ has 33 such convs per 512² face: the encoder levels at 256²/128²/64², and the decoder
resnets and upsamplers from 64² to 512². The S-mode gates pin the CPU device, and the GPU modes carry
no fp32 threshold, so none of this showed.

Every stride-1 3×3 conv is now a `WinogradFreeConv2d`. **Default `.conv3d`** (`model.convRoute`,
type `RestoreFormerConvRoute`).

Measurements: aligned 512² face fixture, production fp32, GPU against the CPU lane.

| | Raw conv2d (Winograd) | conv3d route |
|---|---|---|
| Output | 5.0e-3 · max 0.131 · **17 of 255 levels** | 8.2e-4 · max 1.25e-2 · 2 levels |
| VQ codebook indices | 1 of 256 flipped (a near-tie) | identical |
| Forward time, 512² | 271 ms | **232 ms** — the route is faster at these shapes |

- The remaining 8.2e-4 is TF32 in the attention matmuls. With `MLX_ENABLE_TF32=0` both lanes agree
  to ~5e-6.
- Environment override: `RESTOREFORMER_CONV_ROUTE=winograd|conv3d|fp32Winograd`.
- Gate: `RF_LANE=1 swift test -c release -Xswiftc -enable-testing --filter GPULaneTests`.

## 16 bits end to end (0.3.0)

The surface declares `ToolDescriptor.rawImageFormats = [.rawBGRA8, .rawRGBA16]` (MLXToolKit contract 1.52.0,
AB-T-0230). A `.rawRGBA16` image comes back `.rawRGBA16`; a PNG or TIFF deeper than 8 bits comes back as a 16-bit PNG.
- **Every pixel hop of the face pass runs at 16 bits** (`restoreDeep`). The FFHQ crop is warped into a 16-bit canvas
  (`FaceAlign.warpDeep`), and the restored face is rounded to 16 bits and warped back into one. The feather mask is warped
  at 16 bits (`FaceAlign.warpedMaskDeep`), and the composite runs in float. An 8-bit face inside a 16-bit frame would be
  an 8-bit plateau inside the paste mask, a visible tone step in a sky behind a portrait.
- Faces are found on the image's 8-bit rendering, as the 8-bit path finds them. No face: the input comes back as it was.
- **G-6 gate** (2026-10-12, `mlxengine-image/WIP/g6-16bit-gate`, `faces`): on a portrait in front of a 16-bit sky,
  the 16-bit output's smooth-area flat-neighbour share in the band just inside the paste edge is 0.043, against
  0.90 for the 8-bit output, and the edge shows no step at ×8 contrast. What remains is RestoreFormer++'s own re-render of the
  background inside its crop (about 2 steps darker), feathered, the same at 8 bits.
- 8-bit inputs take the 8-bit path: 62/62 outputs (7 images × png, png at strength 0.5, rawBGRA8, jpeg, with NAFNet and
  the other face model alongside) byte-identical to an output of v0.2.1, in release builds on the GPU.

## Faces in a fixed order (0.3.0)

Vision returns the same faces in no fixed order from run to run, and the paste-back composites them in order. Where two
faces' paste masks overlap, the output depended on the run: across 13 runs of v0.2.1 on GFPGAN's `Blake_Lively`, every
variant came out two ways. `FaceAlign.detectFaces` now sorts faces left to right by the left eye (then top to bottom),
so the output is deterministic and matches one of the two.

## The image edge under a close-up's crop (0.3.1)

When a face's 512² FFHQ crop extended past the frame (every close-up portrait, and faces near an edge), the face pass
pasted a dark 1–2 px line along that image edge (AB-T-0227, FACE-RESTORE-EVAL §7). The input → crop warp filled its
canvas with black, and CoreGraphics antialiases an image's rect boundary against what is underneath, so the image's
outermost pixels came into the crop blended with black. The network kept the dark edge, and the paste mask, 1 there,
wrote it back.

- **The fix is outside the network.** `restore`'s crop warp (8 and 16 bits) now draws the image over its own edge pixels
  extended outward (`FaceAlign.Border.replicate`, `cv2.BORDER_REPLICATE`): each corner pixel fills the quadrant beyond
  its corner, each outermost row or column is stretched across its side, and the image goes on top. Nothing in the crop
  is black where the frame ends. The paste-back, the mask and `warp`'s default (upstream's black border) are unchanged.
  `FACEALIGN_BORDER=black` puts the old crop border back, for parity work.
- **A crop inside the frame is untouched.** The edge layers are drawn only when the crop reaches within a pixel of the
  frame, and even then they lie under the image. Faces whose crop stays inside come out byte-identical to 0.3.0.
- **Measured** (`mlxengine-image/WIP/face-edge-line`, CC0 and public-domain Commons portraits re-cropped so the crop
  leaves each side, both face packages × 8/16 bits × strength 0.5/1.0, release builds on the GPU):
  - outputs with an edge line (outermost two columns ≥ 2× the interior's change + 3 levels): **420 of 524 with
    0.3.0, 0 of 524 with 0.3.1** (131 cases whose crop leaves the frame — 76 faces at 56–142 px between the eyes, 55
    at 170–418 px, the size of a ×4-upscaled close-up — at 8 and 16 bits, strength 0.5 and 1.0; every side);
  - the outermost column's mean change from the input: signed −21 to −25 levels with 0.3.0 (−28 to −31 on the large
    faces), against 4–8 for the interior; −1.9 to −0.2 with 0.3.1, below the interior's own change;
  - in-bounds controls (32 cases): every output byte-identical to 0.3.0. AB-R-0504's whole-image parity rows: 20/20
    identical where no crop leaves the frame; `Blake_Lively` (its crops leave the top and right) changes, as it should;
  - against the same face restored from a wider frame (what the network draws where the frame doesn't end), the fixed
    outermost pixel is closer than keeping the input's outer 2 px would be, so no mask erosion is added.
- Parity gates unchanged (the network is untouched): S0 441/441, S1 7/7, S3 40/40.

## License

Apache-2.0 (port code and weights; upstream is plain Apache-2.0). Residual: ROHQD/model
trained on FFHQ (CC-BY-NC-SA dataset compilation — the standard unsettled question).
