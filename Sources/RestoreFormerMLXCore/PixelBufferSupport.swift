//
//  PixelBufferSupport.swift
//  RestoreFormerMLXCore
//
//  CVPixelBuffer ↔ MLXArray round-trip for full-frame restoration. Same shape and conventions as
//  the sibling image packages (the sibling image packages) — deliberately duplicated rather
//  than cross-depended, since each `-swift` core stays engine- and sibling-agnostic.
//
//  NV12 hazard (forge convention): any byte-level reader must convert to BGRA first — reading NV12
//  base addresses as packed pixels yields sheared garbage.
//

import CoreImage
import CoreVideo
import Foundation
import MLX

/// Convert any `CVPixelBuffer` to 32BGRA via CoreImage (no-op when already BGRA).
public func ensureBGRA(_ input: CVPixelBuffer) -> CVPixelBuffer {
    if CVPixelBufferGetPixelFormatType(input) == kCVPixelFormatType_32BGRA {
        return input
    }
    let w = CVPixelBufferGetWidth(input)
    let h = CVPixelBufferGetHeight(input)
    var out: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: w,
        kCVPixelBufferHeightKey as String: h,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &out) == kCVReturnSuccess,
          let buffer = out else { return input }
    let ctx = CIContext(options: [.cacheIntermediates: false])
    ctx.render(CIImage(cvPixelBuffer: input), to: buffer)
    return buffer
}

/// BGRA `CVPixelBuffer` → `[1, H, W, 3]` RGB float NHWC in [0,1].
public func rgbNHWC(from bgra: CVPixelBuffer, width: Int, height: Int) -> MLXArray? {
    CVPixelBufferLockBaseAddress(bgra, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(bgra, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(bgra) else { return nil }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(bgra)
    let src = base.assumingMemoryBound(to: UInt8.self)

    var rgb = [Float](repeating: 0, count: height * width * 3)
    for y in 0 ..< height {
        let row = y * bytesPerRow
        let drow = y * width * 3
        for x in 0 ..< width {
            let s = row + x * 4
            let d = drow + x * 3
            rgb[d + 0] = Float(src[s + 2]) / 255.0  // R  (BGRA byte 2)
            rgb[d + 1] = Float(src[s + 1]) / 255.0  // G
            rgb[d + 2] = Float(src[s + 0]) / 255.0  // B
        }
    }
    return MLXArray(rgb, [1, height, width, 3])
}

/// `[1, H, W, 3]` RGB float (any range; clamped) → BGRA `CVPixelBuffer`.
public func pixelBuffer(fromRGBNHWC array: MLXArray, width: Int, height: Int) -> CVPixelBuffer? {
    let rgb = array.asArray(Float.self)
    guard rgb.count >= width * height * 3 else { return nil }

    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                              attrs as CFDictionary, &pb) == kCVReturnSuccess,
          let buffer = pb else { return nil }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let dstBase = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let dst = dstBase.assumingMemoryBound(to: UInt8.self)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

    @inline(__always) func clamp(_ v: Float) -> UInt8 {
        UInt8(max(0, min(255, v * 255)))
    }
    for y in 0 ..< height {
        let row = y * bytesPerRow
        let srow = y * width * 3
        for x in 0 ..< width {
            let d = row + x * 4
            let s = srow + x * 3
            dst[d + 0] = clamp(rgb[s + 2])  // B
            dst[d + 1] = clamp(rgb[s + 1])  // G
            dst[d + 2] = clamp(rgb[s + 0])  // R
            dst[d + 3] = 255
        }
    }
    return buffer
}

// MARK: - 16 bits (MLXToolKit contract 1.52.0, AB-T-0230)

/// The 16-bit pixel format: RGBA, 16 bits per component, little-endian — `.rawRGBA16`'s layout, and the shared tile
/// driver's `MLXTileProcessor.deepPixelFormat` in the upscale siblings.
public let deepPixelFormat: OSType = kCVPixelFormatType_64RGBALE

/// Whether `buffer` is a 16-bit (`deepPixelFormat`) buffer.
public func isDeep(_ buffer: CVPixelBuffer) -> Bool {
    CVPixelBufferGetPixelFormatType(buffer) == deepPixelFormat
}

/// RGBA16 `CVPixelBuffer` (`deepPixelFormat`) → `[1, H, W, 3]` RGB float NHWC in [0,1] (each component / 65535).
@inline(never)
public func rgbNHWC(fromRGBA16 buffer: CVPixelBuffer, width: Int, height: Int) -> MLXArray? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    var rgb = [Float](repeating: 0, count: height * width * 3)
    for y in 0 ..< height {
        let src = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt16.self)
        let drow = y * width * 3
        for x in 0 ..< width {
            for c in 0 ..< 3 { rgb[drow + x * 3 + c] = Float(UInt16(littleEndian: src[x * 4 + c])) / 65535.0 }
        }
    }
    return MLXArray(rgb, [1, height, width, 3])
}

/// Interleaved RGBA16 words (tightly packed, `width * 4` per row) → `[1, H, W, 3]` RGB float NHWC in [0,1].
@inline(never)
public func rgbNHWC(fromRGBA16Words words: [UInt16], width: Int, height: Int) -> MLXArray {
    var rgb = [Float](repeating: 0, count: height * width * 3)
    for p in 0 ..< (width * height) {
        for c in 0 ..< 3 { rgb[p * 3 + c] = Float(UInt16(littleEndian: words[p * 4 + c])) / 65535.0 }
    }
    return MLXArray(rgb, [1, height, width, 3])
}

/// `[1, H, W, 3]` RGB float (any range; clamped) → interleaved RGBA16 words, each component rounded to 16 bits, alpha
/// opaque.
@inline(never)
public func rgba16Words(fromRGBNHWC array: MLXArray, width: Int, height: Int) -> [UInt16]? {
    let rgb = array.asType(.float32).asArray(Float.self)
    guard rgb.count >= width * height * 3 else { return nil }
    var words = [UInt16](repeating: UInt16.max.littleEndian, count: width * height * 4)
    for p in 0 ..< (width * height) {
        for c in 0 ..< 3 { words[p * 4 + c] = UInt16(max(0, min(65535, (rgb[p * 3 + c] * 65535).rounded()))).littleEndian }
    }
    return words
}

/// `[1, H, W, 3]` RGB float (any range; clamped) → RGBA16 `CVPixelBuffer` (`deepPixelFormat`), each component rounded
/// to 16 bits, alpha opaque.
@inline(never)
public func pixelBufferRGBA16(fromRGBNHWC array: MLXArray, width: Int, height: Int) -> CVPixelBuffer? {
    guard let words = rgba16Words(fromRGBNHWC: array, width: width, height: height) else { return nil }
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: deepPixelFormat,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    guard CVPixelBufferCreate(nil, width, height, deepPixelFormat, attrs as CFDictionary, &pb) == kCVReturnSuccess,
          let buffer = pb else { return nil }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let dstBase = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    words.withUnsafeBytes { src in
        guard let srcBase = src.baseAddress else { return }
        for y in 0 ..< height {
            memcpy(dstBase.advanced(by: y * bytesPerRow), srcBase.advanced(by: y * width * 8), width * 8)
        }
    }
    return buffer
}
