// GPU-lane gate for mlx's lossy Winograd conv2d window (Sources/RestoreFormerMLXCore/
// WinogradFreeConv2d.swift): the production fp32 forward on a real aligned 512² face, on the CPU
// stream (exact-class reference) and on the GPU with the conv route on (default) and off (raw).
// The S-mode parity gates pin the CPU device and the GPU modes carry no fp32 threshold, so the GPU
// lane was never gated before this.
//
// Measured 2026-09-24 (M5 Max, mlx-swift 0.31.6), face A: raw GPU vs CPU relL2 5.0e-3, max 17 of
// 255 levels, one VQ index flipped; conv3d route 8.2e-4, max 2 levels (the rest is TF32 in the
// attention matmuls); with MLX_ENABLE_TF32=0 both lanes agree to ~5e-6.
//
// Run: RF_LANE=1 swift test -c release -Xswiftc -enable-testing --filter GPULaneTests
// Overrides: RF_WEIGHTS (model.safetensors), RF_FACE (aligned 512² face PNG).

import CoreGraphics
import Foundation
import ImageIO
import MLX
import RestoreFormerMLXCore
import XCTest

final class GPULaneTests: XCTestCase {
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    static func stats(_ a: MLXArray, _ ref: MLXArray) -> (rel: Float, text: String) {
        let d = a.asType(.float32) - ref.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref.asType(.float32))))
        let q = { (x: MLXArray) in clip((x + 1) / 2, min: 0, max: 1) * 255 }  // production truncation
        let lv = abs(floor(q(a)) - floor(q(ref)))
        let mx = abs(d).max()
        eval(rel, mx, lv)
        return (rel.item(Float.self), String(format: "relL2 %.2e  maxAbs %.2e  8-bit max %d levels, %.2f%% px > 2",
            rel.item(Float.self), mx.item(Float.self), Int(lv.max().item(Float.self)),
            100 * mean(lv .> 2).item(Float.self)))
    }

    func testFaceGPUvsCPU() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["RF_LANE"] == "1", "set RF_LANE=1 to run")
        let weights = env["RF_WEIGHTS"]
            ?? "/Volumes/Satechi/Models/models/mlx-community/RestoreFormerPlusPlus-fp32/model.safetensors"
        let face = URL(fileURLWithPath: env["RF_FACE"]
            ?? Self.root.appendingPathComponent("oracle/fixtures/Julia_Roberts_crop.png").path)
        let model = RestoreFormer()
        try model.loadWeights(from: URL(fileURLWithPath: weights))

        guard let src = CGImageSourceCreateWithURL(face as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw NSError(domain: "RF", code: 1) }
        let (w, h) = (cg.width, cg.height)
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        let rgb = (0..<(w * h * 3)).map { i in Float(rgba[(i / 3) * 4 + i % 3]) / 127.5 - 1 }
        let x = MLXArray(rgb, [1, h, w, 3])

        let ref = Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let r = model(x)
            eval(r)
            return r
        }
        Memory.clearCache()
        var t: [RestoreFormerConvRoute: [Double]] = [:]
        var out: [RestoreFormerConvRoute: MLXArray] = [:]
        for _ in 0..<3 {
            for route in [RestoreFormerConvRoute.conv3d, .winograd] {
                model.convRoute = route
                var y = model(x)
                eval(y)
                let t0 = Date()
                for _ in 0..<3 { y = model(x); eval(y) }
                t[route, default: []].append(Date().timeIntervalSince(t0) / 3 * 1000)
                out[route] = y
            }
        }
        model.convRoute = .conv3d
        let sR = Self.stats(out[.conv3d]!, ref), sW = Self.stats(out[.winograd]!, ref)
        func med(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
        print("[\(face.lastPathComponent) \(w)×\(h), fp32, GPU vs CPU lane]")
        print(String(format: "  conv3d route   %@  %6.1f ms", sR.text, med(t[.conv3d]!)))
        print(String(format: "  raw Winograd   %@  %6.1f ms", sW.text, med(t[.winograd]!)))
        if getenv("MLX_ENABLE_TF32").map({ String(cString: $0) }) == "0" {
            XCTAssertLessThan(sR.rel, 1e-4, "conv3d route vs CPU lane (TF32 off)")
        } else {
            // TF32 on (default): the attention matmuls keep ~8e-4 on both paths.
            XCTAssertLessThan(sR.rel, 2e-3, "conv3d route vs CPU lane")
            XCTAssertLessThan(sR.rel, sW.rel / 3, "route vs raw Winograd, against the CPU lane")
        }
    }
}
