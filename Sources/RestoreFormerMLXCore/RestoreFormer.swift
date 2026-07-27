//
//  RestoreFormer.swift
//  mlx-restoreformer-swift / RestoreFormerMLXCore
//
//  Role: MLX-Swift port of RestoreFormer++ (TPAMI 2023) — blind face restoration on an
//        aligned 512×512 crop. VQ-GAN encoder/decoder over a reconstruction-oriented
//        high-quality dictionary (ROHQD, 1024×256 codebook) with multi-scale multi-head
//        cross-attention fusing degraded features into the decode.
//
//  Upstream: https://github.com/wzhouxiff/RestoreFormerPlusPlus — plain Apache-2.0, no
//            third-party carve-outs. 73,472,579 parameters (441 tensors, all model keys).
//  Paper:    Wang et al., "RestoreFormer++", TPAMI 2023.
//
//  Conventions: NHWC; module keys mirror the upstream state dict exactly (`vqvae.` prefix
//  already stripped by oracle/convert.py). Fully deterministic — the VQ lookup is an
//  argmin, there is no noise injection anywhere.
//
//  The one directional subtlety worth stating twice: in `MultiHeadAttnBlock(x, y)` the
//  QUERY is computed from `y` (the encoder feature in the decoder's cross-attention) and
//  K/V from `x`'s normalized stream; the residual is added to `x`. Ported verbatim.
//

import Foundation
import MLX
import MLXNN

@inline(__always) func swish(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

func groupNorm(_ channels: Int) -> GroupNorm {
    GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, affine: true,
              pytorchCompatible: true)
}

/// Nearest-neighbour ×2 upsample (NHWC) — `F.interpolate(scale_factor=2, mode="nearest")`.
func upsampleNearest2x(_ x: MLXArray) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    return broadcast(x.reshaped([b, h, 1, w, 1, c]), to: [b, h, 2, w, 2, c])
        .reshaped([b, h * 2, w * 2, c])
}

/// The VQ bottleneck: nearest codebook entry by L2 (inference path only — no losses).
public final class VectorQuantizer: Module {
    @ModuleInfo(key: "embedding") public var embedding: Embedding

    public init(nEmbed: Int, eDim: Int) {
        self._embedding.wrappedValue = Embedding(embeddingCount: nEmbed, dimensions: eDim)
    }

    /// - Parameter z: `(b, h, w, e_dim)`
    /// - Returns: quantized `(b, h, w, e_dim)` + flat codebook indices `(b·h·w,)`
    public func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray) {
        let e = embedding.weight                                    // (n_e, e_dim)
        let flat = z.reshaped([-1, e.dim(1)])                       // (bhw, e_dim)
        let d = flat.square().sum(axis: 1, keepDims: true)
            + e.square().sum(axis: 1)
            - 2 * matmul(flat, e.transposed(1, 0))                  // (bhw, n_e)
        let indices = argMin(d, axis: 1)                            // (bhw,)
        let zq = take(e, indices, axis: 0).reshaped(z.shape)
        return (zq, indices)
    }
}

/// GroupNorm → swish → conv ×2 with an identity / 1×1 (`nin_shortcut`) skip.
/// (`temb`/`conv_shortcut` are training-config paths no released checkpoint uses.)
public final class ResnetBlock: Module {
    @ModuleInfo(key: "norm1") public var norm1: GroupNorm
    @ModuleInfo(key: "conv1") public var conv1: Conv2d
    @ModuleInfo(key: "norm2") public var norm2: GroupNorm
    @ModuleInfo(key: "conv2") public var conv2: Conv2d
    @ModuleInfo(key: "nin_shortcut") public var ninShortcut: Conv2d?

    public init(inChannels: Int, outChannels: Int) {
        self._norm1.wrappedValue = groupNorm(inChannels)
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._norm2.wrappedValue = groupNorm(outChannels)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._ninShortcut.wrappedValue = inChannels != outChannels
            ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
            : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(swish(norm1(x)))
        h = conv2(swish(norm2(h)))
        let skip = ninShortcut.map { $0(x) } ?? x
        return skip + h
    }
}

/// Spatial multi-head attention over 1×1-conv Q/K/V, channels split head-major.
/// Self-attention when `y == nil`; cross-attention feeds the query from `y`.
public final class MultiHeadAttnBlock: Module {
    public let headSize: Int
    public let attSize: Int

    @ModuleInfo(key: "norm1") public var norm1: GroupNorm
    @ModuleInfo(key: "norm2") public var norm2: GroupNorm
    @ModuleInfo(key: "q") public var q: Conv2d
    @ModuleInfo(key: "k") public var k: Conv2d
    @ModuleInfo(key: "v") public var v: Conv2d
    @ModuleInfo(key: "proj_out") public var projOut: Conv2d

    public init(_ channels: Int, headSize: Int) {
        precondition(channels % headSize == 0)
        self.headSize = headSize
        self.attSize = channels / headSize
        self._norm1.wrappedValue = groupNorm(channels)
        self._norm2.wrappedValue = groupNorm(channels)
        self._q.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self._k.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self._v.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self._projOut.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
    }

    public func callAsFunction(_ x: MLXArray, _ y: MLXArray? = nil) -> MLXArray {
        let hNorm = norm1(x)
        let yIn = y.map { norm2($0) } ?? hNorm

        let (b, hh, ww, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let hw = hh * ww
        // (b,h,w,c) → (b, hw, head, att) — the same head-major channel split as upstream's
        // reshape(b, head, att, h*w); then to (b, head, hw, att) / (b, head, att, hw).
        let qh = q(yIn).reshaped([b, hw, headSize, attSize]).transposed(0, 2, 1, 3)
        let kh = k(hNorm).reshaped([b, hw, headSize, attSize]).transposed(0, 2, 3, 1)
        let vh = v(hNorm).reshaped([b, hw, headSize, attSize]).transposed(0, 2, 1, 3)

        let scale = Float(pow(Double(attSize), -0.5))
        var w_ = softmax(matmul(qh * scale, kh), axis: -1)
        w_ = matmul(w_, vh)                                        // (b, head, hw, att)
        w_ = w_.transposed(0, 2, 1, 3).reshaped([b, hh, ww, c])
        return x + projOut(w_)
    }
}

/// ×2 nearest upsample + 3×3 conv.
public final class Upsample: Module {
    @ModuleInfo(key: "conv") public var conv: Conv2d
    public init(_ channels: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { conv(upsampleNearest2x(x)) }
}

/// Asymmetric zero-pad (right/bottom by 1) + stride-2 3×3 conv — upstream's exact
/// "no asymmetric padding in torch conv, must do it ourselves".
public final class Downsample: Module {
    @ModuleInfo(key: "conv") public var conv: Conv2d
    public init(_ channels: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2, padding: 0)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(padded(x, widths: [IntOrPair(0), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair(0)]))
    }
}

/// One encoder resolution level (`down.N` in the state dict).
public final class DownLevel: Module {
    @ModuleInfo(key: "block") public var block: [ResnetBlock]
    @ModuleInfo(key: "attn") public var attn: [MultiHeadAttnBlock]
    @ModuleInfo(key: "downsample") public var downsample: Downsample?

    init(block: [ResnetBlock], attn: [MultiHeadAttnBlock], downsample: Downsample?) {
        self._block.wrappedValue = block
        self._attn.wrappedValue = attn
        self._downsample.wrappedValue = downsample
    }
}

/// One decoder resolution level (`up.N`).
public final class UpLevel: Module {
    @ModuleInfo(key: "block") public var block: [ResnetBlock]
    @ModuleInfo(key: "attn") public var attn: [MultiHeadAttnBlock]
    @ModuleInfo(key: "upsample") public var upsample: Upsample?

    init(block: [ResnetBlock], attn: [MultiHeadAttnBlock], upsample: Upsample?) {
        self._block.wrappedValue = block
        self._attn.wrappedValue = attn
        self._upsample.wrappedValue = upsample
    }
}

/// The `mid` triple shared by encoder and decoder.
public final class MidBlock: Module {
    @ModuleInfo(key: "block_1") public var block1: ResnetBlock
    @ModuleInfo(key: "attn_1") public var attn1: MultiHeadAttnBlock
    @ModuleInfo(key: "block_2") public var block2: ResnetBlock

    init(_ channels: Int, headSize: Int) {
        self._block1.wrappedValue = ResnetBlock(inChannels: channels, outChannels: channels)
        self._attn1.wrappedValue = MultiHeadAttnBlock(channels, headSize: headSize)
        self._block2.wrappedValue = ResnetBlock(inChannels: channels, outChannels: channels)
    }
}

/// The encoder — returns every intermediate the decoder's cross-attention will query
/// (upstream's `hs` dict, same keys).
public final class MultiHeadEncoder: Module {
    public let numResolutions: Int
    public let numResBlocks: Int

    @ModuleInfo(key: "conv_in") public var convIn: Conv2d
    @ModuleInfo(key: "down") public var down: [DownLevel]
    @ModuleInfo(key: "mid") public var mid: MidBlock
    @ModuleInfo(key: "norm_out") public var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") public var convOut: Conv2d

    public init(ch: Int, chMult: [Int], numResBlocks: Int, attnResolutions: [Int],
                resolution: Int, inChannels: Int, zChannels: Int, headSize: Int) {
        self.numResolutions = chMult.count
        self.numResBlocks = numResBlocks

        self._convIn.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: ch, kernelSize: 3, padding: 1)

        var curRes = resolution
        let inChMult = [1] + chMult
        var levels: [DownLevel] = []
        var blockIn = ch
        for iLevel in 0 ..< chMult.count {
            var blocks: [ResnetBlock] = []
            var attns: [MultiHeadAttnBlock] = []
            blockIn = ch * inChMult[iLevel]
            let blockOut = ch * chMult[iLevel]
            for _ in 0 ..< numResBlocks {
                blocks.append(ResnetBlock(inChannels: blockIn, outChannels: blockOut))
                blockIn = blockOut
                if attnResolutions.contains(curRes) {
                    attns.append(MultiHeadAttnBlock(blockIn, headSize: headSize))
                }
            }
            let ds: Downsample? = iLevel != chMult.count - 1 ? Downsample(blockIn) : nil
            if ds != nil { curRes /= 2 }
            levels.append(DownLevel(block: blocks, attn: attns, downsample: ds))
        }
        self._down.wrappedValue = levels

        self._mid.wrappedValue = MidBlock(blockIn, headSize: headSize)
        self._normOut.wrappedValue = groupNorm(blockIn)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: blockIn, outputChannels: zChannels, kernelSize: 3, padding: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> [String: MLXArray] {
        var hs: [String: MLXArray] = [:]
        var h = convIn(x)
        hs["in"] = h
        for iLevel in 0 ..< numResolutions {
            for iBlock in 0 ..< numResBlocks {
                h = down[iLevel].block[iBlock](h)
                if !down[iLevel].attn.isEmpty {
                    h = down[iLevel].attn[iBlock](h)
                }
            }
            if let ds = down[iLevel].downsample {
                hs["block_\(iLevel)"] = h
                h = ds(h)
            }
            // Per-level graph boundary (same rationale as the decoder's): the monolithic
            // 512² encoder graph otherwise peaks ~3 GB through the engine lane.
            eval(h)
            Memory.clearCache()
        }
        h = mid.block1(h)
        hs["block_\(numResolutions - 1)_atten"] = h
        h = mid.attn1(h)
        h = mid.block2(h)
        hs["mid_atten"] = h
        hs["out"] = convOut(swish(normOut(h)))
        return hs
    }
}

/// The decoder with multi-scale cross-attention into the encoder's `hs` features.
public final class MultiHeadDecoderTransformer: Module {
    public let numResolutions: Int
    public let numResBlocks: Int

    @ModuleInfo(key: "conv_in") public var convIn: Conv2d
    @ModuleInfo(key: "mid") public var mid: MidBlock
    @ModuleInfo(key: "up") public var up: [UpLevel]
    @ModuleInfo(key: "norm_out") public var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") public var convOut: Conv2d

    public init(ch: Int, outCh: Int, chMult: [Int], numResBlocks: Int, attnResolutions: [Int],
                resolution: Int, zChannels: Int, headSize: Int) {
        self.numResolutions = chMult.count
        self.numResBlocks = numResBlocks

        var blockIn = ch * chMult[chMult.count - 1]
        var curRes = resolution / (1 << (chMult.count - 1))

        self._convIn.wrappedValue = Conv2d(
            inputChannels: zChannels, outputChannels: blockIn, kernelSize: 3, padding: 1)
        self._mid.wrappedValue = MidBlock(blockIn, headSize: headSize)

        var levels: [UpLevel] = []
        for iLevel in (0 ..< chMult.count).reversed() {
            var blocks: [ResnetBlock] = []
            var attns: [MultiHeadAttnBlock] = []
            let blockOut = ch * chMult[iLevel]
            for _ in 0 ..< (numResBlocks + 1) {
                blocks.append(ResnetBlock(inChannels: blockIn, outChannels: blockOut))
                blockIn = blockOut
                if attnResolutions.contains(curRes) {
                    attns.append(MultiHeadAttnBlock(blockIn, headSize: headSize))
                }
            }
            let us: Upsample? = iLevel != 0 ? Upsample(blockIn) : nil
            if us != nil { curRes *= 2 }
            levels.insert(UpLevel(block: blocks, attn: attns, upsample: us), at: 0)
        }
        self._up.wrappedValue = levels

        self._normOut.wrappedValue = groupNorm(blockIn)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: blockIn, outputChannels: outCh, kernelSize: 3, padding: 1)
    }

    public func callAsFunction(_ z: MLXArray, hs: [String: MLXArray],
                               tap: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        var h = convIn(z)
        h = mid.block1(h)
        h = mid.attn1(h, hs["mid_atten"])
        h = mid.block2(h)
        tap?("dec_mid", h)
        for iLevel in (0 ..< numResolutions).reversed() {
            for iBlock in 0 ..< (numResBlocks + 1) {
                h = up[iLevel].block[iBlock](h)
                if !up[iLevel].attn.isEmpty {
                    h = up[iLevel].attn[iBlock](h, hs["block_\(iLevel)_atten"] ?? hs["block_\(iLevel)"])
                }
            }
            if let us = up[iLevel].upsample {
                h = us(h)
            }
            tap?("dec_level\(iLevel)", h)
            // Per-level graph boundary — the GFPGAN lesson: one monolithic 512² decode
            // graph balloons the transient working set and strands dirty driver pages.
            eval(h)
            Memory.clearCache()
        }
        return convOut(swish(normOut(h)))
    }
}

/// RestoreFormer++ — the full VQ-GAN + multi-scale cross-attention restorer.
public final class RestoreFormer: Module, @unchecked Sendable {

    public struct Configuration: Sendable {
        public var nEmbed = 1024
        public var embedDim = 256
        public var ch = 64
        public var outCh = 3
        public var chMult = [1, 2, 2, 4, 4, 8]
        public var numResBlocks = 2
        /// Encoder attention resolutions. The decoder ADDS one scale per
        /// `exMultiScaleNum` (the ++ change): enc @16 → dec @[16, 32].
        public var attnResolutions = [16]
        public var inChannels = 3
        public var resolution = 512
        public var zChannels = 256
        public var headSize = 4
        public var exMultiScaleNum = 1

        /// The release `RestoreFormer++.ckpt` config is exactly these defaults.
        public init() {}
    }

    public let configuration: Configuration

    @ModuleInfo(key: "encoder") public var encoder: MultiHeadEncoder
    @ModuleInfo(key: "decoder") public var decoder: MultiHeadDecoderTransformer
    @ModuleInfo(key: "quantize") public var quantize: VectorQuantizer
    @ModuleInfo(key: "quant_conv") public var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") public var postQuantConv: Conv2d

    public init(_ cfg: Configuration = Configuration()) {
        self.configuration = cfg

        self._encoder.wrappedValue = MultiHeadEncoder(
            ch: cfg.ch, chMult: cfg.chMult, numResBlocks: cfg.numResBlocks,
            attnResolutions: cfg.attnResolutions, resolution: cfg.resolution,
            inChannels: cfg.inChannels, zChannels: cfg.zChannels, headSize: cfg.headSize)

        var decAttn = cfg.attnResolutions
        for _ in 0 ..< cfg.exMultiScaleNum {
            decAttn = [decAttn[0], decAttn[decAttn.count - 1] * 2]
        }
        self._decoder.wrappedValue = MultiHeadDecoderTransformer(
            ch: cfg.ch, outCh: cfg.outCh, chMult: cfg.chMult, numResBlocks: cfg.numResBlocks,
            attnResolutions: decAttn, resolution: cfg.resolution,
            zChannels: cfg.zChannels, headSize: cfg.headSize)

        self._quantize.wrappedValue = VectorQuantizer(nEmbed: cfg.nEmbed, eDim: cfg.embedDim)
        self._quantConv.wrappedValue = Conv2d(
            inputChannels: cfg.zChannels, outputChannels: cfg.embedDim, kernelSize: 1)
        self._postQuantConv.wrappedValue = Conv2d(
            inputChannels: cfg.embedDim, outputChannels: cfg.zChannels, kernelSize: 1)
    }

    /// Restore an aligned face crop.
    ///
    /// - Parameters:
    ///   - x: `(b, 512, 512, 3)` RGB in **[-1, 1]** (`(img/255 - 0.5) / 0.5`).
    ///   - tap: gate-only observer, called with the oracle's golden names per stage.
    /// - Returns: `(b, 512, 512, 3)` RGB in [-1, 1], unclamped (clamp at the consumer).
    public func callAsFunction(_ x: MLXArray,
                               tap: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        let hs = encoder(x)
        if let tap { for (k, v) in hs { tap("enc_\(k)", v) } }
        let h = quantConv(hs["out"]!)
        tap?("quant_conv", h)
        let (zq, indices) = quantize(h)
        tap?("zq", zq)
        tap?("indices", indices)
        let q = postQuantConv(zq)
        tap?("post_quant", q)
        return decoder(q, hs: hs, tap: tap)
    }

    /// Loads converted safetensors weights under the strict verifier.
    public func loadWeights(from url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
        try update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        eval(self)
    }
}
