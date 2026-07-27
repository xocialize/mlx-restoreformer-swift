//
//  main.swift
//  mlx-restoreformer-swift / RestoreFormerValidate
//
//  Drives the package through the **real `MLXServeEngine`** and reports the authoritative
//  split footprint (`MLXEngineTestKit.ValidationHarness`, 150 ms `phys_footprint` sampling,
//  floor read post-load/pre-run) — the number the manifest declares. The gate's `--bench`
//  bypasses register/prepare and the governor; this lane does not.
//
//  A face-restoration validate needs a REAL FACE in the input — a synthetic gradient would
//  measure the no-face passthrough, not the forward. Default input is the oracle's face
//  fixture; pass any photo to measure multi-face behavior. The restored PNG is written next
//  to the input (suffix `.restoreformer.png`) for the eyeball check.
//
//  Usage:  swift run restoreformer-validate <weights.safetensors> [image.png] [strength]
//

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXServeCore
import MLXEngineTestKit
import MLXRestoreFormer

setvbuf(stdout, nil, _IONBF, 0)

let args = Array(CommandLine.arguments.dropFirst())
guard let weightsPath = args.first else {
    print("usage: restoreformer-validate <weights.safetensors> [image.png] [strength]")
    exit(2)
}
let imagePath = args.count > 1 ? args[1] : "oracle/goldens/full_face_in.png"
let strength = args.count > 2 ? Float(args[2]) : nil

func makeImage() -> MLXToolKit.Image {
    guard let data = FileManager.default.contents(atPath: imagePath) else {
        print("❌ cannot read \(imagePath) — a face-restoration validate needs a face image")
        exit(2)
    }
    let src = CGImageSourceCreateWithData(data as CFData, nil)
    let cg = src.flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }
    print("input: \(imagePath) (\(cg?.width ?? 0)x\(cg?.height ?? 0))")
    return MLXToolKit.Image(format: .png, data: data, width: cg?.width, height: cg?.height)
}

@MainActor
func main() async {
    // Match the shipping configuration: Forge constructs the engine `.blocking`, so a package
    // that would be refused in production must be refused here too.
    let engine = MLXServeEngine(policy: .permissiveOnly, licenseEnforcement: .blocking)

    let config = RestoreFormerConfiguration(weightsURL: URL(fileURLWithPath: weightsPath))
    let request = ImageRestoreRequest(image: makeImage(), strength: strength)

    do {
        let result = try await ValidationHarness.run(
            engine: engine,
            registration: RestoreFormerRestorePackage.registration,
            configuration: config,
            capability: .imageRestore,
            request: request,
            isolate: true,
            clearCache: { MLX.Memory.clearCache() },
            inputSummary: imagePath,
            heartbeatLabel: "restoreformer")

        print("")
        print(result.run.splitLogLine("restoreformer-\(config.variant.rawValue)"))
        print("")
        print("  DECLARE  residentBytes        = \(result.run.residentFloorBytes)")
        print("           peakActivationBytes  = \(result.run.activationBytes)")
        if result.run.retainedAfterRunBytes > 200_000_000 {
            print("  ⚠️ retains \(result.run.retainedAfterRunBytes) B after run+clearCache — "
                + "a live model holding intermediates. Belongs in the transient, not residency.")
        }
        if let out = result.response as? ImageRestoreResponse {
            let applied = out.appliedStrength.map { String(format: "%.2f", $0) } ?? "nil (no face)"
            print("  output: \(out.image.data.count) bytes, "
                + "\(out.image.width ?? 0)x\(out.image.height ?? 0), appliedStrength=\(applied)")
            let outPath = imagePath.replacingOccurrences(of: ".png", with: "") + ".restoreformer.png"
            try? out.image.data.write(to: URL(fileURLWithPath: outPath))
            print("  wrote \(outPath) — eyeball it")
        }
    } catch {
        print("❌ validation failed: \(error)")
        exit(1)
    }
}

await main()
