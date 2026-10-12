// DeepBoundaryTests.swift — the 16-bit boundary (contract 1.52.0, AB-T-0230), offline: the codec, and the 16-bit warp
// and paste mask that keep a face's region at the frame's depth. CoreGraphics and CoreVideo only, no MLX kernels.

import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import MLXToolKit
import UniformTypeIdentifiers
import XCTest
import RestoreFormerMLXCore
@testable import MLXRestoreFormer

final class DeepBoundaryTests: XCTestCase {

    /// A 16-bit gradient: ~0.1 of an 8-bit step per pixel in red, so an 8-bit hop would collapse it.
    static func deepWords(_ w: Int, _ h: Int) -> [UInt16] {
        var words = [UInt16](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 4
                words[i] = UInt16(20_000 + x * 25); words[i + 1] = UInt16(10_000 + y * 7)
                words[i + 2] = UInt16((x * 977 + y * 131) % 65_536); words[i + 3] = UInt16.max
            }
        }
        return words
    }

    static func deepImage(_ w: Int, _ h: Int) throws -> CGImage {
        try XCTUnwrap(RestoreFormerRestorePackage.deepCGImage(words: deepWords(w, h), width: w, height: h))
    }

    func testSurfaceDeclaresTheSixteenBitBoundary() {
        let surface = RestoreFormerRestorePackage.manifest.surfaces[0]
        XCTAssertEqual(surface.rawImageFormats, [.rawBGRA8, .rawRGBA16])
        XCTAssertTrue(surface.acceptsRawRGBA16)
    }

    func testRawRGBA16RoundTripsBitIdentical() throws {
        let (w, h) = (9, 5)
        let bytes = Self.deepWords(w, h).withUnsafeBytes { Data($0) }
        let pb = try RestoreFormerRestorePackage.decodeToPixelBuffer(.rawRGBA16(data: bytes, width: w, height: h))
        XCTAssertEqual(CVPixelBufferGetPixelFormatType(pb), deepPixelFormat)
        let back = try XCTUnwrap(RestoreFormerRestorePackage.encodeRawRGBA16(pb))
        XCTAssertEqual(back.format, .rawRGBA16)
        XCTAssertEqual(back.data, bytes)
    }

    func testRawRGBA16HonorsSourceStride() throws {
        let (w, h) = (3, 2)
        let words = Self.deepWords(w, h)
        var strided = Data(count: 32 * h)      // 24 bytes of pixels + 8 of padding per row
        for row in 0..<h {
            let rowBytes = words[(row * w * 4)..<((row + 1) * w * 4)].withUnsafeBytes { Data($0) }
            strided.replaceSubrange((row * 32)..<(row * 32 + 24), with: rowBytes)
        }
        let pb = try RestoreFormerRestorePackage.decodeToPixelBuffer(.rawRGBA16(data: strided, width: w, height: h, bytesPerRow: 32))
        let back = try XCTUnwrap(RestoreFormerRestorePackage.encodeRawRGBA16(pb))
        XCTAssertEqual(back.data, words.withUnsafeBytes { Data($0) })
    }

    func testRawRGBA16WithoutDimensionsOrBytesThrows() {
        XCTAssertThrowsError(try RestoreFormerRestorePackage.decodeToPixelBuffer(Image(format: .rawRGBA16, data: Data(count: 64))))
        XCTAssertThrowsError(try RestoreFormerRestorePackage.decodeToPixelBuffer(.rawRGBA16(data: Data(count: 8), width: 2, height: 2)))
    }

    /// A 16-bit PNG decodes at 16 bits (exactly, for an opaque sRGB source) and encodes back to a 16-bit PNG.
    func testDeepPNGDecodesAndEncodesAtSixteenBits() throws {
        let (w, h) = (16, 8)
        let out = NSMutableData()
        let dest = try XCTUnwrap(CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, try Self.deepImage(w, h), nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        let pb = try RestoreFormerRestorePackage.decodeToPixelBuffer(Image(format: .png, data: out as Data, width: w, height: h))
        XCTAssertEqual(CVPixelBufferGetPixelFormatType(pb), deepPixelFormat)
        XCTAssertEqual(try XCTUnwrap(RestoreFormerRestorePackage.encodeRawRGBA16(pb)).data,
                       Self.deepWords(w, h).withUnsafeBytes { Data($0) })
        let png = try XCTUnwrap(RestoreFormerRestorePackage.encodePNG(pb))
        let cg = try XCTUnwrap(CGImageSourceCreateWithData(png as CFData, nil).flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) })
        XCTAssertEqual(cg.bitsPerComponent, 16)
    }

    /// An 8-bit image keeps the 8-bit path.
    func testEightBitRawKeepsTheEightBitPath() throws {
        let pb = try RestoreFormerRestorePackage.decodeToPixelBuffer(.rawBGRA8(data: Data(repeating: 128, count: 64), width: 4, height: 4))
        XCTAssertEqual(CVPixelBufferGetPixelFormatType(pb), kCVPixelFormatType_32BGRA)
    }

    /// The crop warp at 16 bits keeps sub-8-bit precision through a rotation and scale, and draws what the 8-bit warp
    /// draws (within the 8-bit warp's rounding).
    func testWarpDeepKeepsSixteenBits() throws {
        let (w, h) = (300, 200)
        let source = try Self.deepImage(w, h)
        let t = CGAffineTransform(a: 1.4, b: 0.2, c: -0.2, d: 1.4, tx: -40, ty: -30)
        let deep = try XCTUnwrap(FaceAlign.warpDeep(source, transform: t, width: 256, height: 256))
        let eightBitSource = try XCTUnwrap(RestoreFormerRestorePackage.makePixelBuffer(from: source).flatMap(RestoreFormerRestorePackage.cgImage(from:)))
        let eight = try XCTUnwrap(FaceAlign.warp(eightBitSource, transform: t, width: 256, height: 256))
        let eightPB = try XCTUnwrap(RestoreFormerRestorePackage.makePixelBuffer(from: eight))
        let eightRaw = try XCTUnwrap(RestoreFormerRestorePackage.encodeRawBGRA8(eightPB)).data
        var offGrid = 0, interior = 0, worst = 0.0
        for y in 40..<216 {
            for x in 40..<216 {
                let p = y * 256 + x
                let red = deep[p * 4]
                guard red > 0 else { continue }                         // inside the source, not the border
                interior += 1
                if red % 257 != 0 { offGrid += 1 }
                worst = max(worst, abs(Double(red) / 257 - Double(eightRaw[p * 4 + 2])))
            }
        }
        XCTAssertGreaterThan(interior, 10_000)
        XCTAssertGreaterThan(Double(offGrid) / Double(interior), 0.9)   // 16-bit values, not the 8-bit grid
        XCTAssertLessThanOrEqual(worst, 1.5)                            // the same warp, ±the 8-bit route's rounding
    }

    /// The paste mask at 16 bits: the same feathered shape as the 8-bit mask, off its 1/255 grid. (CoreGraphics weights
    /// its interpolation in subpixel steps, so a warped 27-level feather has a few hundred levels at either depth.)
    func testWarpedMaskDeepMatchesTheEightBitMask() throws {
        let t = CGAffineTransform(a: 2.1, b: 0.3, c: -0.3, d: 2.1, tx: -300, ty: -200)
        let deep = try XCTUnwrap(FaceAlign.warpedMaskDeep(transform: t, width: 400, height: 300))
        let eight = try XCTUnwrap(FaceAlign.warpedMask(transform: t, width: 400, height: 300))
        XCTAssertEqual(deep.count, eight.count)
        var worst: Float = 0
        for i in 0..<deep.count { worst = max(worst, abs(deep[i] - eight[i])) }
        XCTAssertLessThan(worst, 3.0 / 255)
        let ramp = deep.filter { $0 > 0.01 && $0 < 0.99 }
        let offGrid = ramp.filter { abs($0 * 255 - ($0 * 255).rounded()) > 0.01 }
        XCTAssertGreaterThan(ramp.count, 1_000)
        XCTAssertGreaterThan(Double(offGrid.count) / Double(ramp.count), 0.5)
        XCTAssertGreaterThan(Set(deep.map { Int(($0 * 65535).rounded()) }).count,
                             Set(eight.map { Int(($0 * 255).rounded()) }).count)
        XCTAssertEqual(deep.max() ?? 0, 1, accuracy: 1e-4)
        XCTAssertEqual(deep.min() ?? 1, 0, accuracy: 1e-4)
    }
}
