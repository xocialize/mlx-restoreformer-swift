//
//  FaceAlign.swift
//  mlx-restoreformer-swift / MLXRestoreFormer
//  (verbatim from mlx-gfpgan-swift, where the Vision-vs-facexlib agreement gate lives —
//   same FaceRestoreHelper template, same face_size=512, so the gate covers both consumers)
//
//  The detect → align → paste-back seam, on Apple Vision instead of facexlib.
//
//  Upstream (`facexlib.FaceRestoreHelper`) runs RetinaFace-ResNet50 for detection + 5-point
//  landmarks, `cv2.estimateAffinePartial2D` for the similarity transform onto the FFHQ
//  template, and a ParseNet soft mask for paste-back. All three are auxiliary signals
//  consumed UPSTREAM of the restoration weights — a replaceable preprocessing seam, not
//  architecture. Substitutions here:
//    detection/landmarks  → VNDetectFaceLandmarksRequest (5 points derived below)
//    estimateAffinePartial2D → closed-form least-squares similarity (same objective,
//                              minus LMedS robustness — moot for 5 points)
//    ParseNet soft mask   → feathered border mask warped with the same inverse affine
//                           (facexlib's own use_parse=False fallback shape)
//
//  The substituted alignment is a distribution shift vs the crops GFPGAN was trained on, so
//  it is GATED, not assumed: mlx-gfpgan-swift/oracle/gen_align_fixtures.py bakes facexlib's affine matrices
//  and crops once offline, and the align gate compares this implementation against them
//  (crop IoU / template-point distance) before the seam is trusted.
//

import CoreGraphics
import CoreImage
import Foundation
import Vision

/// One detected face: five landmarks in image coordinates (top-left origin) and the
/// similarity transform mapping image → 512² FFHQ-template crop.
public struct AlignedFace {
    /// [left eye, right eye, nose tip, left mouth corner, right mouth corner], image px,
    /// top-left origin, ordered by x within each pair (viewer's left first).
    public let landmarks: [CGPoint]
    /// Image coords (top-left) → 512² crop coords.
    public let transform: CGAffineTransform
}

public enum FaceAlignError: Error, Equatable {
    case detectionFailed(String)
    case renderFailed
}

public enum FaceAlign {

    /// The FFHQ 5-point template for a 512² crop (facexlib `FaceRestoreHelper.face_template`):
    /// left eye, right eye, nose tip, left mouth corner, right mouth corner.
    public static let template512: [CGPoint] = [
        CGPoint(x: 192.98138, y: 239.94708),
        CGPoint(x: 318.90277, y: 240.1936),
        CGPoint(x: 256.63416, y: 314.01935),
        CGPoint(x: 201.26117, y: 371.41043),
        CGPoint(x: 313.08905, y: 371.15118),
    ]

    public static let cropSize = 512

    // MARK: - Detection

    /// Detect faces and derive the 5-point landmark set, mirroring upstream's
    /// `get_face_landmarks_5(eye_dist_threshold: 5)` filter.
    public static func detectFaces(in image: CGImage,
                                   eyeDistThreshold: CGFloat = 5) throws -> [AlignedFace] {
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        do { try handler.perform([request]) }
        catch { throw FaceAlignError.detectionFailed(String(describing: error)) }

        let size = CGSize(width: image.width, height: image.height)
        var faces: [AlignedFace] = []
        for obs in request.results ?? [] {
            guard let lm = obs.landmarks else { continue }
            guard let five = fivePoints(from: lm, imageSize: size) else { continue }
            let eyeDist = hypot(five[1].x - five[0].x, five[1].y - five[0].y)
            guard eyeDist >= eyeDistThreshold else { continue }
            let t = similarityTransform(from: five, to: template512)
            faces.append(AlignedFace(landmarks: five, transform: t))
        }
        return faces
    }

    /// Vision landmark regions → the RetinaFace-style 5 points, in top-left image coords.
    /// Eye/mouth pairs are ordered by x, so Vision's left/right naming convention is moot.
    static func fivePoints(from lm: VNFaceLandmarks2D, imageSize: CGSize) -> [CGPoint]? {
        func imagePoints(_ region: VNFaceLandmarkRegion2D?) -> [CGPoint] {
            guard let region else { return [] }
            return region.pointsInImage(imageSize: imageSize)
                .map { CGPoint(x: $0.x, y: imageSize.height - $0.y) }   // flip to top-left
        }
        func mean(_ pts: [CGPoint]) -> CGPoint? {
            guard !pts.isEmpty else { return nil }
            let n = CGFloat(pts.count)
            return CGPoint(x: pts.reduce(0) { $0 + $1.x } / n,
                           y: pts.reduce(0) { $0 + $1.y } / n)
        }

        guard let eyeA = imagePoints(lm.leftPupil).first ?? mean(imagePoints(lm.leftEye)),
              let eyeB = imagePoints(lm.rightPupil).first ?? mean(imagePoints(lm.rightEye))
        else { return nil }
        let nose = imagePoints(lm.noseCrest).last ?? mean(imagePoints(lm.nose))
        guard let noseTip = nose else { return nil }
        let lips = imagePoints(lm.outerLips)
        guard let mouthA = lips.min(by: { $0.x < $1.x }),
              let mouthB = lips.max(by: { $0.x < $1.x }) else { return nil }

        let eyes = eyeA.x <= eyeB.x ? [eyeA, eyeB] : [eyeB, eyeA]
        let mouth = [mouthA, mouthB]   // min/max by x already ordered
        return [eyes[0], eyes[1], noseTip, mouth[0], mouth[1]]
    }

    // MARK: - Similarity transform

    /// Least-squares non-reflective similarity `dst ≈ s·R·src + t` — the same objective as
    /// `cv2.estimateAffinePartial2D` without the LMedS sampling (with exactly 5 well-spread
    /// points the robust estimator degenerates to least squares anyway).
    ///
    /// Parameterized `x' = a·x − b·y + tx`, `y' = b·x + a·y + ty`; the normal equations are
    /// linear in `(a, b, tx, ty)` and solved in closed form.
    public static func similarityTransform(from src: [CGPoint], to dst: [CGPoint])
        -> CGAffineTransform {
        precondition(src.count == dst.count && src.count >= 2)
        let n = CGFloat(src.count)
        var sx: CGFloat = 0, sy: CGFloat = 0, su: CGFloat = 0, sv: CGFloat = 0
        var sxx: CGFloat = 0, sux: CGFloat = 0, svx: CGFloat = 0
        for (p, q) in zip(src, dst) {
            sx += p.x; sy += p.y; su += q.x; sv += q.y
            sxx += p.x * p.x + p.y * p.y
            sux += q.x * p.x + q.y * p.y          // Σ(u·x + v·y)
            svx += q.y * p.x - q.x * p.y          // Σ(v·x − u·y)
        }
        let d = sxx - (sx * sx + sy * sy) / n
        let a = (sux - (sx * su + sy * sv) / n) / d
        let b = (svx - (sx * sv - sy * su) / n) / d
        let tx = (su - a * sx + b * sy) / n
        let ty = (sv - b * sx - a * sy) / n
        // CGAffineTransform is row-vector: [x y 1]·[a b; c d; tx ty]
        return CGAffineTransform(a: a, b: b, c: -b, d: a, tx: tx, ty: ty)
    }

    // MARK: - Warping

    /// Warp `image` by `transform` (top-left coords) into a `width`×`height` canvas.
    /// Bilinear-quality interpolation via CoreGraphics; the black border matches upstream's
    /// `cv2.warpAffine` zero border.
    public static func warp(_ image: CGImage, transform: CGAffineTransform,
                            width: Int, height: Int) -> CGImage? {
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.interpolationQuality = .high

        // The transform is authored in top-left coordinates; CG draws bottom-left. Conjugate
        // with the two flips: C = F_src · T · F_dst (row-vector composition, F_src applied first).
        let fSrc = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: CGFloat(image.height))
        let fDst = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: CGFloat(height))
        ctx.concatenate(fSrc.concatenating(transform).concatenating(fDst))
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return ctx.makeImage()
    }

    /// The feathered paste-back mask in crop space: 1 in the interior, linear ramp to 0 over
    /// `feather` px at the borders — the shape of facexlib's use_parse=False erosion+blur mask.
    /// Returned as row-major floats in [0, 1], `cropSize`².
    public static func featherMask(feather: Int = 26) -> [Float] {
        let n = cropSize
        var mask = [Float](repeating: 1, count: n * n)
        let f = max(1, feather)
        for y in 0 ..< n {
            for x in 0 ..< n {
                let d = min(min(x, n - 1 - x), min(y, n - 1 - y))
                if d < f { mask[y * n + x] = Float(d) / Float(f) }
            }
        }
        return mask
    }

    /// Warp the crop-space feather mask into image space (zeros outside the face region).
    public static func warpedMask(transform cropFromImage: CGAffineTransform,
                                  width: Int, height: Int,
                                  feather: Int = 26) -> [Float]? {
        let n = cropSize
        let mask8 = featherMask(feather: feather).map { UInt8($0 * 255) }
        guard let provider = CGDataProvider(data: Data(mask8) as CFData),
              let gray = CGImage(width: n, height: n, bitsPerComponent: 8, bitsPerPixel: 8,
                                 bytesPerRow: n, space: CGColorSpaceCreateDeviceGray(),
                                 bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider,
                                 decode: nil, shouldInterpolate: true, intent: .defaultIntent),
              let warped = warp8bitGray(gray, transform: cropFromImage.inverted(),
                                        width: width, height: height) else { return nil }
        return warped.map { Float($0) / 255 }
    }

    private static func warp8bitGray(_ image: CGImage, transform: CGAffineTransform,
                                     width: Int, height: Int) -> [UInt8]? {
        var data = [UInt8](repeating: 0, count: width * height)
        let ok: Bool = data.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(
                data: buf.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo(rawValue: 0).rawValue) else { return false }
            ctx.interpolationQuality = .high
            let fSrc = CGAffineTransform(a: 1, b: 0, c: 0, d: -1,
                                         tx: 0, ty: CGFloat(image.height))
            let fDst = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: CGFloat(height))
            ctx.concatenate(fSrc.concatenating(transform).concatenating(fDst))
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            return true
        }
        return ok ? data : nil
    }
}
