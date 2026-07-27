"""RestoreFormer++ oracle — per-sub-op goldens for the Swift port.

fp32, CPU-torch, numpy-seeded, C-contiguous, PyTorch NCHW. The Swift gate transposes to
NHWC, runs, transposes back, compares.

Fully deterministic — no noise injection anywhere (the VQ lookup is an argmin). The one
parity-fragile spot is codebook argmin near-ties, so the goldens include the raw indices:
the gate reports exact index agreement alongside the z_q tensor tolerance.

The full-model decoder taps come from a hand-replicated decoder loop; the encoder needs no
replica (its forward already returns every intermediate in the `hs` dict). The replica is
SELF-CHECKING: its final image is asserted bit-equal against a direct `model(x)[0]` call.

Run:  .venv/bin/python gen_goldens.py
Out:  goldens/*.npy  +  goldens/MANIFEST.txt  +  goldens/*.png (eyeball)
"""
import importlib.util
import os
import sys

import numpy as np
import torch
from PIL import Image

torch.set_grad_enabled(False)

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "vqvae_arch", os.path.join(HERE, "upstream", "RestoreFormer", "modules", "vqvae", "vqvae_arch.py"))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

raw = torch.load(os.path.join(HERE, "weights", "RestoreFormer++.ckpt"),
                 map_location="cpu", weights_only=False)
stripped = {k.replace("vqvae.", "", 1): v for k, v in raw["state_dict"].items()
            if k.startswith("vqvae.")}
model = A.VQVAEGANMultiHeadTransformer()
missing, _ = model.load_state_dict(stripped, strict=False)
assert not missing
model.eval()

OUT = os.path.join(HERE, "goldens")
os.makedirs(OUT, exist_ok=True)
manifest = []


def save(name, arr, dtype=np.float32):
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    np.save(os.path.join(OUT, name + ".npy"), a)
    manifest.append(f"{name + '.npy':40s} {str(a.shape):26s} dtype={a.dtype}")
    print(f"  saved {name}.npy  {tuple(a.shape)}")


def dump(name, t):
    save(name, t.detach().cpu().numpy())


def seeded(seed, *shape):
    g = np.random.default_rng(seed)
    return torch.from_numpy(g.standard_normal(shape, dtype=np.float32))


print("=== 1. Components ===")
xr = seeded(6001, 1, 64, 64, 64)
dump("resblock_nin_in", xr)
dump("resblock_nin_out", model.encoder.down[1].block[0](xr, None))     # 64 -> 128, nin_shortcut

xs = seeded(6002, 1, 64, 32, 32)
dump("resblock_same_in", xs)
dump("resblock_same_out", model.encoder.down[0].block[0](xs, None))    # 64 -> 64, identity skip

xa = seeded(6003, 1, 512, 16, 16)
dump("selfattn_in", xa)
dump("selfattn_out", model.encoder.mid.attn_1(xa))

ya = seeded(6004, 1, 512, 16, 16)
dump("crossattn_y", ya)
dump("crossattn_out", model.decoder.mid.attn_1(xa, ya))

dump("downsample_in", xs)
dump("downsample_out", model.encoder.down[0].downsample(xs))           # asym pad + stride-2

xu = seeded(6005, 1, 512, 8, 8)
dump("upsample_in", xu)
dump("upsample_out", model.decoder.up[5].upsample(xu))                 # nearest x2 + conv

zq_in = seeded(6006, 1, 256, 16, 16)
zq, _, info = model.quantize(zq_in)
dump("quantize_in", zq_in)
dump("quantize_out", zq)
save("quantize_indices", info[2].numpy().reshape(-1), dtype=np.int64)

print("\n=== 2. Full model — encoder hs + replicated decoder taps (self-checked) ===")

ENC_KEYS = ["in", "block_0", "block_1", "block_2", "block_3", "block_4",
            "block_5_atten", "mid_atten", "out"]


def full_taps(tag, x):
    dump(f"{tag}_in", x)

    hs = model.encoder(x)
    assert set(hs.keys()) == set(ENC_KEYS), sorted(hs.keys())
    for k in ENC_KEYS:
        dump(f"{tag}_enc_{k}", hs[k])

    h = model.quant_conv(hs["out"])
    dump(f"{tag}_quant_conv", h)
    quant, _, info = model.quantize(h)
    dump(f"{tag}_zq", quant)
    save(f"{tag}_indices", info[2].numpy().reshape(-1), dtype=np.int64)

    # replicated MultiHeadDecoderTransformer.forward with per-level taps
    dec = model.decoder
    q = model.post_quant_conv(quant)
    dump(f"{tag}_post_quant", q)
    h = dec.conv_in(q)
    h = dec.mid.block_1(h, None)
    h = dec.mid.attn_1(h, hs["mid_atten"])
    h = dec.mid.block_2(h, None)
    dump(f"{tag}_dec_mid", h)
    for i_level in reversed(range(dec.num_resolutions)):
        for i_block in range(dec.num_res_blocks + 1):
            h = dec.up[i_level].block[i_block](h, None)
            if len(dec.up[i_level].attn) > 0:
                if f"block_{i_level}_atten" in hs:
                    h = dec.up[i_level].attn[i_block](h, hs[f"block_{i_level}_atten"])
                else:
                    h = dec.up[i_level].attn[i_block](h, hs[f"block_{i_level}"])
        if i_level != 0:
            h = dec.up[i_level].upsample(h)
        dump(f"{tag}_dec_level{i_level}", h)
    h = dec.norm_out(h)
    h = A.nonlinearity(h)
    image = dec.conv_out(h)
    dump(f"{tag}_image", image)

    direct = model(x)[0]
    assert torch.equal(image, direct), "replicated decoder diverged from model(x)[0]!"
    print(f"  ✅ replica == model(x)[0] bit-exact ({tag})")
    return image


g = np.random.default_rng(7100)
x_rand = torch.from_numpy((g.random((1, 3, 512, 512), dtype=np.float32) * 2 - 1))
full_taps("full_rand", x_rand)

face = Image.open(os.path.join(HERE, "fixtures", "Julia_Roberts_crop.png"))
face = face.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
face_np = np.asarray(face, dtype=np.float32) / 255.0
x_face = torch.from_numpy(np.ascontiguousarray(
    ((face_np - 0.5) / 0.5).transpose(2, 0, 1)))[None]
img = full_taps("full_face", x_face)

for tag, t in (("full_face_in", x_face), ("full_face_out", img)):
    arr = t[0].numpy().transpose(1, 2, 0)
    arr = np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT, tag + ".png"))
print("  saved eyeball PNGs (full_face_in/out)")

with open(os.path.join(OUT, "MANIFEST.txt"), "w") as f:
    f.write("RestoreFormer++ PyTorch goldens — fp32, CPU, PyTorch NCHW, C-contiguous.\n")
    f.write("checkpoint: weights/RestoreFormer++.ckpt (state_dict, vqvae.* stripped)\n")
    f.write("constructor: VQVAEGANMultiHeadTransformer() — the ++ defaults ARE the release\n")
    f.write("  config: ch=64 ch_mult=(1,2,2,4,4,8) n_embed=1024 embed_dim=256 head_size=4\n")
    f.write("  enc attn@16, dec attn@[16,32] (ex_multi_scale_num=1)\n")
    f.write("input contract: RGB, 512x512, [-1,1]; output [-1,1] clamp; fully deterministic\n\n")
    f.write("\n".join(manifest) + "\n")

print(f"\n✅ {len(manifest)} goldens written to {OUT}/")
