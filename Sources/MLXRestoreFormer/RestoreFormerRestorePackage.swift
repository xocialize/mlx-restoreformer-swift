import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXProfiling
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
///
/// **16 bits end to end** (contract 1.52.0, AB-T-0230): the surface declares `.rawRGBA16`. A `.rawRGBA16` image, or a
/// PNG/TIFF deeper than 8 bits, takes `restoreDeep`: the same detect → align → restore → paste-back with every pixel hop
/// at 16 bits and the composite in float, so the face's region keeps the frame's depth. It comes back at 16 bits:
/// `.rawRGBA16` for raw, a 16-bit PNG otherwise. 8-bit inputs take the 8-bit path byte for byte.
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
                // Mirrors rawBGRA8 (since 0.1.0) and keeps 16 bits for rawRGBA16 and deep PNG (0.3.0).
                .declaring(rawImageFormats: [.rawBGRA8, .rawRGBA16])
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
            // The engine materialized the declared `weightSources` before load() (contract 1.24): `StoreWeights` reads
            // its copy, and downloads through `HubApi` only for a caller outside the engine.
            url = try await StoreWeights.resolve(repo: configuration.variant.repo, root: configuration.modelsRootDirectory)
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
        if isDeep(pb) { return try restoreDeep(pb, request: req, model: model, strength: strength) }
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

    /// The 16-bit route (AB-T-0230): `run`'s detect → align → restore → paste-back with every pixel hop at 16 bits. The
    /// input is read at 16 bits, the FFHQ crop is warped into a 16-bit canvas, the restored face is rounded to 16 bits and
    /// warped back into a 16-bit canvas, the feather mask is warped at 16 bits, and the composite runs in float. An 8-bit
    /// face inside a 16-bit frame would be an 8-bit plateau inside the paste mask and a tone step at its edge. Faces are
    /// found on the image's 8-bit rendering, as the 8-bit route finds them: detection needs no more depth.
    private func restoreDeep(_ pb: CVPixelBuffer, request req: ImageRestoreRequest, model: RestoreFormer,
                             strength: Float) throws -> ImageRestoreResponse {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let source = Self.deepCGImage(pb),
              let eightBit = Self.makePixelBuffer(from: source).flatMap(Self.cgImage(from:)) else {
            throw RestoreFormerPackageError.imageDecodeFailed("CGImage conversion (\(w)x\(h), 16-bit)")
        }
        let faces: [AlignedFace]
        do {
            faces = try FaceAlign.detectFaces(in: eightBit,
                                              eyeDistThreshold: CGFloat(configuration.minEyeDistance))
        } catch {
            throw RestoreFormerPackageError.faceDetectionFailed(String(describing: error))
        }
        try Task.checkCancellation()
        guard !faces.isEmpty else {
            return ImageRestoreResponse(image: req.image, appliedStrength: nil)
        }
        guard var canvas = rgbNHWC(fromRGBA16: pb, width: w, height: h) else {
            throw RestoreFormerPackageError.imageDecodeFailed("NHWC conversion (\(w)x\(h), 16-bit)")
        }

        let prof = MLXProfiler.shared
        prof.beginRun("restoreformer imageRestore \(configuration.variant.rawValue) "
            + "\(w)x\(h) faces=\(faces.count) 16-bit")
        let n = FaceAlign.cropSize
        for (idx, face) in faces.enumerated() {
            try Task.checkCancellation()
            RunProgress.report(RunPhaseReport(phase: .postprocess, step: idx + 1,
                                              totalSteps: faces.count))
            guard let cropWords = FaceAlign.warpDeep(source, transform: face.transform, width: n, height: n) else {
                throw RestoreFormerPackageError.imageDecodeFailed("face crop warp (face \(idx), 16-bit)")
            }
            let crop = rgbNHWC(fromRGBA16Words: cropWords, width: n, height: n)
            let restored = try prof.region("restore", "forward") { () -> MLXArray in
                let out = model(crop * 2 - 1)
                return clip((out + 1) / 2, min: 0, max: 1)
            }
            let blended = strength == 1 ? restored : restored * strength + crop * (1 - strength)
            guard let blendedWords = rgba16Words(fromRGBNHWC: blended, width: n, height: n),
                  let blendedCG = Self.deepCGImage(words: blendedWords, width: n, height: n),
                  let faceWords = FaceAlign.warpDeep(blendedCG, transform: face.transform.inverted(),
                                                     width: w, height: h),
                  let maskFlat = FaceAlign.warpedMaskDeep(transform: face.transform,
                                                         width: w, height: h,
                                                         feather: configuration.pasteFeather) else {
                throw RestoreFormerPackageError.imageEncodeFailed
            }
            let faceNHWC = rgbNHWC(fromRGBA16Words: faceWords, width: w, height: h)
            let mask = MLXArray(maskFlat, [1, h, w, 1])
            canvas = faceNHWC * mask + canvas * (1 - mask)
            eval(canvas)
        }
        prof.endRun(denominators: ["face": Double(faces.count)])

        try Task.checkCancellation()
        guard let outPB = pixelBufferRGBA16(fromRGBNHWC: canvas, width: w, height: h) else {
            throw RestoreFormerPackageError.imageEncodeFailed
        }
        let outImage: Image
        if req.image.format == .rawRGBA16 {
            guard let raw = Self.encodeRawRGBA16(outPB) else { throw RestoreFormerPackageError.imageEncodeFailed }
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

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8/.rawRGBA16) to a `CVPixelBuffer`: BGRA, or 16-bit RGBA
    /// (`deepPixelFormat`) for `.rawRGBA16` and for an encoded image deeper than 8 bits per component.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        if image.format == .rawRGBA16 { return try rawRGBA16ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RestoreFormerPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        if cg.bitsPerComponent > 8 { return try deepPixelBuffer(drawing: cg) }
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

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes (a 16-bit PNG for a `deepPixelFormat` buffer).
    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        if isDeep(pb) {
            guard let cg = deepCGImage(pb) else { return nil }
            let out = NSMutableData()
            guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
            CGImageDestinationAddImage(dest, cg, nil)
            return CGImageDestinationFinalize(dest) ? out as Data : nil
        }
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

    // MARK: - 16 bits (contract 1.52.0, AB-T-0230; the mlx-realplksr-swift 0.2.0 codec)

    /// RGBA, 16 bits per component, little-endian, premultiplied — `.rawRGBA16`'s layout as CoreGraphics draws it.
    nonisolated static let rgba16BitmapInfo = FaceAlign.rgba16BitmapInfo

    nonisolated static func makeDeepPixelBuffer(_ w: Int, _ h: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: deepPixelFormat,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, deepPixelFormat, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RestoreFormerPackageError.imageDecodeFailed("16-bit pixel buffer allocation (\(w)x\(h))")
        }
        return buffer
    }

    /// Draw `cg` into a new `deepPixelFormat` buffer, in sRGB at 16 bits.
    nonisolated static func deepPixelBuffer(drawing cg: CGImage) throws -> CVPixelBuffer {
        let w = cg.width, h = cg.height
        let buffer = try makeDeepPixelBuffer(w, h)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 16,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: rgba16BitmapInfo) else {
            throw RestoreFormerPackageError.imageDecodeFailed("CGContext for RGBA16 draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Wrap raw interleaved RGBA16 bytes straight into a `deepPixelFormat` buffer — no decode.
    nonisolated static func rawRGBA16ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw RestoreFormerPackageError.imageDecodeFailed("rawRGBA16 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 8)
        guard srcStride >= w * 8, image.data.count >= srcStride * h else {
            throw RestoreFormerPackageError.imageDecodeFailed(
                "rawRGBA16 data too small (\(image.data.count) < \(srcStride * h))")
        }
        let buffer = try makeDeepPixelBuffer(w, h)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw RestoreFormerPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), w * 8)
            }
        }
        return buffer
    }

    /// A `deepPixelFormat` buffer as a 16-bit sRGB `CGImage` over a copy of its pixels.
    nonisolated static func deepCGImage(_ pb: CVPixelBuffer) -> CGImage? {
        guard let raw = encodeRawRGBA16(pb) else { return nil }
        return deepCGImage(data: raw.data, width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
    }

    /// Interleaved RGBA16 words (tightly packed) as a 16-bit sRGB `CGImage`.
    nonisolated static func deepCGImage(words: [UInt16], width: Int, height: Int) -> CGImage? {
        deepCGImage(data: words.withUnsafeBytes { Data($0) }, width: width, height: height)
    }

    nonisolated static func deepCGImage(data: Data, width: Int, height: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 16, bitsPerPixel: 64, bytesPerRow: width * 8,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: rgba16BitmapInfo),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// Emit a `deepPixelFormat` buffer as tightly-packed `.rawRGBA16` bytes.
    nonisolated static func encodeRawRGBA16(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0, isDeep(pb) else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 8
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawRGBA16(data: out, width: w, height: h)
    }
}

extension RestoreFormerRestorePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(RestoreFormerRestorePackage.self)
    }
}
