// EdgeBorderTests.swift — the crop warp at the frame edge (AB-T-0227), offline: CoreGraphics only, no MLX kernels.
// A face whose 512² crop extends past the frame had the image's outermost pixels blended with black inside the crop, and
// the paste wrote that back as a dark 1–2 px line along the image edge. The crop warp now replicates the edge pixels.

import CoreGraphics
import Foundation
import XCTest
@testable import MLXRestoreFormer

final class EdgeBorderTests: XCTestCase {

    /// A 320×240 photo stand-in, every value in 90...200: smooth ramps and a fine texture, so a black border shows as a
    /// value under 90.
    static func value(_ x: Int, _ y: Int, _ c: Int) -> Double {
        90 + 55 * Double(x) / 320 + 40 * Double(y) / 240 + 7.5 * (1 + sin(Double(x * (c + 2) + y * 3) * 0.15))
    }

    static func image8(_ w: Int = 320, _ h: Int = 240) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: w * h * 4)
        for y in 0..<h { for x in 0..<w { for c in 0..<3 { bytes[(y * w + x) * 4 + c] = UInt8(value(x, y, c).rounded()) } } }
        return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: CGDataProvider(data: Data(bytes) as CFData)!, decode: nil, shouldInterpolate: true,
                       intent: .defaultIntent)!
    }

    static func image16(_ w: Int = 320, _ h: Int = 240) -> CGImage {
        var words = [UInt16](repeating: .max, count: w * h * 4)
        for y in 0..<h { for x in 0..<w { for c in 0..<3 { words[(y * w + x) * 4 + c] = UInt16((value(x, y, c) * 257).rounded()) } } }
        return RestoreFormerRestorePackage.deepCGImage(words: words.map { $0.littleEndian }, width: w, height: h)!
    }

    /// The image → crop transform of a face at (cx, cy) with the given inter-ocular distance and roll: the FFHQ template
    /// mapped into the image, then fitted back, as `detectFaces` does from landmarks.
    static func faceTransform(cx: CGFloat, cy: CGFloat, iod: CGFloat, roll: CGFloat) -> CGAffineTransform {
        let s = iod / (FaceAlign.template512[1].x - FaceAlign.template512[0].x)
        let toImage = CGAffineTransform(translationX: -256, y: -280)
            .concatenating(CGAffineTransform(rotationAngle: roll)).concatenating(CGAffineTransform(scaleX: s, y: s))
            .concatenating(CGAffineTransform(translationX: cx, y: cy))
        return FaceAlign.similarityTransform(from: FaceAlign.template512.map { $0.applying(toImage) },
                                             to: FaceAlign.template512)
    }

    /// Near the top-left corner, rolled 8°: the crop leaves the frame on the left and at the top.
    static let leaving = faceTransform(cx: 70, cy: 75, iod: 60, roll: 0.14)
    /// Centred and small: the crop stays inside with room.
    static let inside = faceTransform(cx: 160, cy: 125, iod: 40, roll: -0.1)

    /// RGB of a BGRA (8-bit warp) canvas, in 8-bit levels.
    static func rgb(_ image: CGImage) -> [Double] {
        let (w, h) = (image.width, image.height)
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        bytes.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        var out = [Double](); out.reserveCapacity(w * h * 3)
        for i in 0..<(w * h) { for c in 0..<3 { out.append(Double(bytes[i * 4 + c])) } }
        return out
    }

    static func rgb(words: [UInt16]) -> [Double] {
        var out = [Double](); out.reserveCapacity(words.count / 4 * 3)
        for i in 0..<(words.count / 4) { for c in 0..<3 { out.append(Double(UInt16(littleEndian: words[i * 4 + c])) / 257) } }
        return out
    }

    func testTheLeavingTransformLeavesTheFrame() {
        let inv = Self.leaving.inverted()
        let corners = [CGPoint(x: 0, y: 0), CGPoint(x: 512, y: 0), CGPoint(x: 512, y: 512), CGPoint(x: 0, y: 512)].map { $0.applying(inv) }
        XCTAssertLessThan(corners.map(\.x).min()!, -20)
        XCTAssertLessThan(corners.map(\.y).min()!, -20)
    }

    /// `.black` puts black in the crop where the frame ends; `.replicate` puts nothing darker than the image's darkest.
    func testReplicateLeavesNoBlackInTheCrop() throws {
        let black = Self.rgb(try XCTUnwrap(FaceAlign.warp(Self.image8(), transform: Self.leaving, width: 512, height: 512)))
        let repl = Self.rgb(try XCTUnwrap(FaceAlign.warp(Self.image8(), transform: Self.leaving, width: 512, height: 512,
                                                         border: .replicate)))
        XCTAssertEqual(black.min(), 0)
        XCTAssertGreaterThanOrEqual(repl.min()!, 89)
        XCTAssertLessThanOrEqual(repl.max()!, 201)

        let deep = Self.rgb(words: try XCTUnwrap(FaceAlign.warpDeep(Self.image16(), transform: Self.leaving, width: 512,
                                                                     height: 512, border: .replicate)))
        XCTAssertGreaterThanOrEqual(deep.min()!, 89.5)
        XCTAssertEqual(Self.rgb(words: try XCTUnwrap(FaceAlign.warpDeep(Self.image16(), transform: Self.leaving,
                                                                         width: 512, height: 512))).min(), 0)
    }

    /// A crop inside the frame, or reaching to within a pixel of it, is `.black`'s byte for byte, at 8 and 16 bits.
    func testInsideTheFrameReplicateIsTheBlackWarp() throws {
        // Inside by half a pixel: the edge layers are drawn but lie under the image or outside the canvas.
        let inv = Self.inside.inverted()
        let minX = [CGPoint(x: 0, y: 0), CGPoint(x: 512, y: 0), CGPoint(x: 512, y: 512), CGPoint(x: 0, y: 512)]
            .map { $0.applying(inv).x }.min()!
        let nearEdge = CGAffineTransform(translationX: minX - 0.5, y: 0).concatenating(Self.inside)
        for t in [Self.inside, nearEdge] {
            let a = try XCTUnwrap(FaceAlign.warp(Self.image8(), transform: t, width: 512, height: 512))
            let b = try XCTUnwrap(FaceAlign.warp(Self.image8(), transform: t, width: 512, height: 512, border: .replicate))
            XCTAssertEqual(a.dataProvider?.data as Data?, b.dataProvider?.data as Data?)
            XCTAssertEqual(FaceAlign.warpDeep(Self.image16(), transform: t, width: 512, height: 512),
                           FaceAlign.warpDeep(Self.image16(), transform: t, width: 512, height: 512, border: .replicate))
        }
    }

    /// `restore`'s paste-back with the network replaced by the identity: crop, warp back, feathered mask, composite. With
    /// `.black` the outermost column comes back dark — the defect; with `.replicate` it departs from the input no more
    /// than the columns inside it, which only the two resamplings touch.
    func testIdentityRestorerKeepsTheImageEdge() throws {
        let (w, h) = (320, 240)
        let src = Self.image8()
        let input = Self.rgb(src)
        let mask = try XCTUnwrap(FaceAlign.warpedMask(transform: Self.leaving, width: w, height: h))
        func leftProfile(_ border: FaceAlign.Border) throws -> (outermost: Double, interior: Double) {
            let crop = try XCTUnwrap(FaceAlign.warp(src, transform: Self.leaving, width: 512, height: 512, border: border))
            let back = Self.rgb(try XCTUnwrap(FaceAlign.warp(crop, transform: Self.leaving.inverted(), width: w, height: h)))
            var d0 = 0.0, inner = 0.0, n = 0.0
            for y in 0..<h where mask[y * w] > 0.99 {
                func delta(_ x: Int) -> Double {
                    var t = 0.0
                    for c in 0..<3 {
                        let i = (y * w + x) * 3, m = Double(mask[y * w + x])
                        t += abs(back[i + c] * m + input[i + c] * (1 - m) - input[i + c])
                    }
                    return t / 3
                }
                d0 += delta(0); for x in 4...12 { inner += delta(x) / 9 }; n += 1
            }
            XCTAssertGreaterThan(n, 50)
            return (d0 / n, inner / n)
        }
        let black = try leftProfile(.black), repl = try leftProfile(.replicate)
        XCTAssertGreaterThan(black.outermost, 20)                       // the dark line
        XCTAssertLessThan(repl.outermost, repl.interior + 1.5)          // gone
        XCTAssertLessThan(repl.outermost, 3)
    }
}
