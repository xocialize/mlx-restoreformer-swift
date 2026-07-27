import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXProfiling
import Hub
import RestoreFormerMLXCore

/// Errors at the GFPGAN package boundary.
public enum RestoreFormerPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
    case weightsMissing(String)
    case faceDetectionFailed(String)
}

/// An MLXEngine `imageRestore` package over **RestoreFormer++** (TPAMI 2023) — blind FACE
/// restoration from a learned high-quality dictionary.
///
/// The sixth package on `imageRestore`, and the SECOND face restorer alongside GFPGAN —
/// PackageID-selected redundancy on the same job, not a quality tier: GFPGAN restores from
/// a StyleGAN2 generative prior (stronger hallucination, stronger identity drift risk);
/// RestoreFormer++ reconstructs through a VQ codebook of real HQ facial features with
/// cross-attention to the degraded input (more conservative, better fidelity on heavy
/// degradation per its TPAMI numbers). Same Vision detect → FFHQ-align → restore →
/// feathered paste-back pipeline; non-face pixels pass through untouched.
///
/// `strength` IS honoured (contract 1.30.0): it blends the restored face with the original
/// aligned crop before paste-back — the package-level dial both face restorers share.
///
/// No face found ⇒ the input passes through unchanged with `appliedStrength == nil`.
@InferenceActor
public final class RestoreFormerRestorePackage: ModelPackage {
    public typealias Configuration = RestoreFormerConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // wzhouxiff/RestoreFormerPlusPlus is plain Apache-2.0 — no third-party
            // carve-outs at all (the favourable case; contrast GFPGAN's NVIDIA/DFDNet
            // clauses). Weights are the author's own v1.0.0 GitHub release; the ROHQD
            // codebook is trained on FFHQ (dataset compilation CC-BY-NC-SA; the standard
            // unsettled dataset→weights question that applies to every face model).
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(sourceRepo: "wzhouxiff/RestoreFormerPlusPlus",
                                   revision: "v1.0.0", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint — ✅ MEASURED through the REAL `MLXServeEngine` via
                // `MLXEngineTestKit.ValidationHarness` (`swift run restoreformer-validate`),
                // process `phys_footprint`, floor read post-load/pre-run:
                //
                //   [restoreformer-plusPlus] SPLIT floor=0.31GB peak=1.79GB act=1.47GB
                //                            retain=0.67GB load=0.0s run=1.8s @1297x1920, 2 faces
                //
                // Declared with margin: resident 350 MB (floor 310.5 MB), activation 2.0 GB
                // (measured 1.47 GB). Fixed-size forward (every face at exactly 512²) —
                // flat in input resolution. The per-level eval + clearCache seams in BOTH
                // encoder and decoder are load-bearing: without the encoder's, the same run
                // measured act 2.98 GB through this lane — do not remove them casually.
                footprints: [
                    QuantFootprint(quant: .fp32,
                                   residentBytes: 350_000_000,
                                   peakActivationBytes: 2_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageRestoreContract.descriptor(
                    name: "restoreformer-face-restore",
                    summary: "RestoreFormer++ blind face restoration: detects and aligns "
                        + "every face, reconstructs each through a learned HQ facial "
                        + "codebook with cross-attention (more conservative than a GAN "
                        + "prior), and pastes back. Non-face regions pass through. strength "
                        + "blends restored vs original face (1 = full restoration).",
                    supportsStrength: true
                )
            ]
        )
    }

    private let configuration: Configuration
    private var model: RestoreFormer?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }

        let url: URL
        if let explicit = configuration.weightsURL {
            guard FileManager.default.fileExists(atPath: explicit.path) else {
                throw RestoreFormerPackageError.weightsMissing(explicit.path)
            }
            url = explicit
        } else {
            // Since contract 1.24 the engine materializes declared `weightSources` before
            // load(); this snapshot is the defensive path for engine-less consumers.
            let repo = configuration.variant.repo
            let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
            let dir = try await hub.snapshot(from: Hub.Repo(id: repo),
                                             matching: ["model.safetensors"]) { progress, speed in
                WeightDownloadProgress.report(fraction: progress.fractionCompleted, bytesPerSecond: speed)
            }
            url = dir.appendingPathComponent("model.safetensors")
        }

        let net = RestoreFormer()
        try net.loadWeights(from: url)
        model = net
    }

    public func unload() async {
        model = nil
        MLX.Memory.clearCache()   // drop the retained MLX pool so eviction frees RSS, not just refs
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint is the FIRST act of run(), before notLoaded validation.
        // Mid-run cadence: one checkpoint per detected face (the real iterative seam), plus
        // the post-detect and pre-encode phase boundaries.
        try Task.checkCancellation()
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .imageRestore,
              let req = request as? ImageRestoreRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let strength = max(0, min(1, req.strength ?? 1))

        let pb = try Self.decodeToPixelBuffer(req.image)
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let bgra = ensureBGRA(pb)
        guard let cg = Self.cgImage(from: bgra) else {
            throw RestoreFormerPackageError.imageDecodeFailed("CGImage conversion (\(w)x\(h))")
        }

        // Detect + align (Vision, CPU) — cheap relative to the forward.
        let faces: [AlignedFace]
        do {
            faces = try FaceAlign.detectFaces(in: cg,
                                              eyeDistThreshold: CGFloat(configuration.minEyeDistance))
        } catch {
            throw RestoreFormerPackageError.faceDetectionFailed(String(describing: error))
        }
        try Task.checkCancellation()

        // No face: the restoration is a no-op, honestly reported (appliedStrength nil).
        guard !faces.isEmpty else {
            return ImageRestoreResponse(image: req.image, appliedStrength: nil)
        }

        guard var canvas = rgbNHWC(from: bgra, width: w, height: h) else {
            throw RestoreFormerPackageError.imageDecodeFailed("NHWC conversion (\(w)x\(h))")
        }

        let prof = MLXProfiler.shared
        prof.beginRun("restoreformer imageRestore \(configuration.variant.rawValue) "
            + "\(w)x\(h) faces=\(faces.count)")
        let n = FaceAlign.cropSize
        for (idx, face) in faces.enumerated() {
            // Per-face checkpoint — the CAN-3 cadence unit.
            try Task.checkCancellation()
            RunProgress.report(RunPhaseReport(phase: .postprocess, step: idx + 1,
                                              totalSteps: faces.count))

            guard let cropCG = FaceAlign.warp(cg, transform: face.transform,
                                              width: n, height: n),
                  let cropPB = Self.makePixelBuffer(from: cropCG),
                  let crop = rgbNHWC(from: ensureBGRA(cropPB), width: n, height: n) else {
                throw RestoreFormerPackageError.imageDecodeFailed("face crop warp (face \(idx))")
            }

            // [0,1] → [-1,1], restore, clamp back — GFPGANer's normalize/tensor2img pair.
            let restored = try prof.region("restore", "forward") { () -> MLXArray in
                let out = model(crop * 2 - 1)
                return clip((out + 1) / 2, min: 0, max: 1)
            }
            // strength: blend restored vs original aligned crop BEFORE paste-back.
            let blended = strength == 1 ? restored : restored * strength + crop * (1 - strength)

            // Paste back: warp crop → image space with the inverse affine + feathered mask.
            guard let blendedPB = RestoreFormerMLXCore.pixelBuffer(fromRGBNHWC: blended, width: n, height: n),
                  let blendedCG = Self.cgImage(from: blendedPB),
                  let faceFull = FaceAlign.warp(blendedCG, transform: face.transform.inverted(),
                                                width: w, height: h),
                  let faceFullPB = Self.makePixelBuffer(from: faceFull),
                  let faceNHWC = rgbNHWC(from: ensureBGRA(faceFullPB), width: w, height: h),
                  let maskFlat = FaceAlign.warpedMask(transform: face.transform,
                                                     width: w, height: h,
                                                     feather: configuration.pasteFeather) else {
                throw RestoreFormerPackageError.imageEncodeFailed
            }
            let mask = MLXArray(maskFlat, [1, h, w, 1])
            canvas = faceNHWC * mask + canvas * (1 - mask)
            eval(canvas)
        }
        prof.endRun(denominators: ["face": Double(faces.count)])

        // Post-forward checkpoint: between materialization and output encode.
        try Task.checkCancellation()
        guard let outPB = RestoreFormerMLXCore.pixelBuffer(fromRGBNHWC: canvas, width: w, height: h) else {
            throw RestoreFormerPackageError.imageEncodeFailed
        }
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw RestoreFormerPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw RestoreFormerPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageRestoreResponse(image: outImage, appliedStrength: strength)
    }

    // MARK: - Image codec
    //
    // Same shape as the sibling image packages. Duplicated rather than shared: each `-swift`
    // package stays independently buildable, and the codec is the package's own boundary.

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8) to a BGRA `CVPixelBuffer`.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RestoreFormerPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RestoreFormerPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw RestoreFormerPackageError.imageDecodeFailed("CGContext for BGRA draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// BGRA `CVPixelBuffer` → `CGImage`.
    nonisolated static func cgImage(from pb: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        return ctx.createCGImage(ci, from: ci.extent)
    }

    /// `CGImage` → BGRA `CVPixelBuffer`.
    nonisolated static func makePixelBuffer(from cg: CGImage) -> CVPixelBuffer? {
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes.
    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    /// Wrap raw interleaved BGRA8 bytes straight into a 32BGRA `CVPixelBuffer` — no decode.
    nonisolated static func rawBGRA8ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw RestoreFormerPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 4)
        guard srcStride >= w * 4, image.data.count >= srcStride * h else {
            throw RestoreFormerPackageError.imageDecodeFailed(
                "rawBGRA8 data too small (\(image.data.count) < \(srcStride * h))")
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RestoreFormerPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw RestoreFormerPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = min(srcStride, dstStride)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), rowBytes)
            }
        }
        return buffer
    }

    /// Emit a 32BGRA `CVPixelBuffer` as tightly-packed raw BGRA8 `Image` bytes.
    nonisolated static func encodeRawBGRA8(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 4
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawBGRA8(data: out, width: w, height: h)
    }
}

extension RestoreFormerRestorePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(RestoreFormerRestorePackage.self)
    }
}
