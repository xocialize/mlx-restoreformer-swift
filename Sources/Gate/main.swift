//
//  main.swift
//  mlx-restoreformer-swift / RestoreFormerGate
//
//  Parity gates against the PyTorch oracle. Executable, not a test target — the SPM test
//  product's metallib is unreliable for GPU work.
//
//  Fully deterministic model (no noise anywhere); the codebook argmin is the one
//  tie-sensitive spot, so S3 gates the flat INDICES on exact equality alongside the float
//  tensors — a near-tie flip would surface as a nonzero mismatch count, localized.
//

import Foundation
import RestoreFormerMLXCore
import MLX
import MLXNN

private let _unbuffered: Void = { setvbuf(stdout, nil, _IONBF, 0) }()

func fail(_ msg: String) -> Never { _ = _unbuffered; print("❌ \(msg)"); exit(1) }

func loadedModel(_ path: String) -> RestoreFormer {
    let model = RestoreFormer()
    do { try model.loadWeights(from: URL(fileURLWithPath: path)) }
    catch { fail("weight load failed: \(error)") }
    return model
}

func g(_ dir: String, _ name: String) -> MLXArray {
    do { return try loadNPY("\(dir)/\(name).npy") } catch { fail("golden \(name): \(error)") }
}

/// int64 .npy loader for the codebook indices (Goldens.swift's reader is deliberately
/// fp32-only, so the indices go through their own strict path).
func loadIndices(_ path: String) -> [Int64] {
    guard let data = FileManager.default.contents(atPath: path) else { fail("cannot read \(path)") }
    let headerLen = Int(data[8]) | (Int(data[9]) << 8)
    guard let header = String(data: data[10 ..< 10 + headerLen], encoding: .ascii),
          header.contains("'descr': '<i8'") else { fail("\(path): expected int64 indices") }
    let body = data[(10 + headerLen)...]
    return body.withUnsafeBytes { raw in
        Array(raw.bindMemory(to: Int64.self))
    }
}

func gateS0(_ weightsPath: String) {
    _ = _unbuffered
    print("=== S0 · key contract ===\n")
    let model = RestoreFormer()
    var swiftKeys: [String: [Int]] = [:]; var total = 0
    for (k, v) in model.parameters().flattened() { swiftKeys[k] = v.shape; total += v.size }
    print("Swift module tree : \(swiftKeys.count) tensors, \(total) params")
    guard let loaded = try? MLX.loadArrays(url: URL(fileURLWithPath: weightsPath)) else {
        fail("could not load \(weightsPath)")
    }
    print("Checkpoint        : \(loaded.count) tensors, \(loaded.values.reduce(0) { $0 + $1.size }) params\n")
    let sk = Set(swiftKeys.keys), ck = Set(loaded.keys)
    let missing = sk.subtracting(ck).sorted(), unused = ck.subtracting(sk).sorted()
    if !missing.isEmpty { print("MISSING (\(missing.count)):"); missing.prefix(15).forEach { print("   \($0)  \(swiftKeys[$0]!)") } }
    if !unused.isEmpty { print("UNUSED (\(unused.count)):"); unused.prefix(15).forEach { print("   \($0)  \(loaded[$0]!.shape)") } }
    var mismatch: [(String, [Int], [Int])] = []
    for k in sk.intersection(ck) where swiftKeys[k]! != loaded[k]!.shape {
        mismatch.append((k, swiftKeys[k]!, loaded[k]!.shape))
    }
    if !mismatch.isEmpty {
        print("SHAPE MISMATCH (\(mismatch.count)):")
        for (k, a, b) in mismatch.prefix(15) { print("   \(k)\n     swift \(a) vs ckpt \(b)") }
    }
    guard missing.isEmpty, unused.isEmpty, mismatch.isEmpty else { fail("S0 FAILED") }
    do { try model.update(parameters: ModuleParameters.unflattened(loaded), verify: .all) }
    catch { fail("S0 FAILED at update(verify: .all): \(error)") }
    print("✅ S0 PASSED — \(swiftKeys.count) tensors, \(total) params, strict update clean.")
}

/// Components: the VQGAN blocks + the VQ lookup itself.
func gateS1(_ dir: String, _ w: String) -> Bool {
    print("=== S1 · components ===\n")
    let r = GateReport("S1")
    let model = loadedModel(w)

    r.check("resblock_nin", toNCHW(model.encoder.down[1].block[0](toNHWC(g(dir, "resblock_nin_in")))),
            g(dir, "resblock_nin_out"), tol: 1e-5)
    r.check("resblock_same", toNCHW(model.encoder.down[0].block[0](toNHWC(g(dir, "resblock_same_in")))),
            g(dir, "resblock_same_out"), tol: 1e-5)
    r.check("selfattn", toNCHW(model.encoder.mid.attn1(toNHWC(g(dir, "selfattn_in")))),
            g(dir, "selfattn_out"), tol: 1e-5)
    r.check("crossattn",
            toNCHW(model.decoder.mid.attn1(toNHWC(g(dir, "selfattn_in")),
                                           toNHWC(g(dir, "crossattn_y")))),
            g(dir, "crossattn_out"), tol: 1e-5)
    r.check("downsample", toNCHW(model.encoder.down[0].downsample!(toNHWC(g(dir, "downsample_in")))),
            g(dir, "downsample_out"), tol: 1e-5)
    r.check("upsample", toNCHW(model.decoder.up[5].upsample!(toNHWC(g(dir, "upsample_in")))),
            g(dir, "upsample_out"), tol: 1e-5)

    let (zq, idx) = model.quantize(toNHWC(g(dir, "quantize_in")))
    r.check("quantize_zq", toNCHW(zq), g(dir, "quantize_out"), tol: 1e-5)
    let want = loadIndices("\(dir)/quantize_indices.npy")
    let got = idx.asArray(Int64.self)
    let mismatches = zip(got, want).filter { $0 != $1 }.count
    print("  \(mismatches == 0 ? "✅" : "❌") quantize_indices      \(want.count - mismatches)/\(want.count) exact")
    return r.summarize() && mismatches == 0
}

/// Full model with per-stage taps — every intermediate the oracle dumped, both inputs.
func gateS3(_ dir: String, _ w: String) -> Bool {
    print("=== S3 · full model (per-stage taps) ===\n")
    let r = GateReport("S3")
    let model = loadedModel(w)
    var indicesOK = true

    for tag in ["full_rand", "full_face"] {
        print("  — \(tag) —")
        var taps: [String: MLXArray] = [:]
        let image = model(toNHWC(g(dir, "\(tag)_in"))) { name, arr in taps[name] = arr }
        eval(image)

        var names = ["enc_in", "enc_block_0", "enc_block_1", "enc_block_2", "enc_block_3",
                     "enc_block_4", "enc_block_5_atten", "enc_mid_atten", "enc_out",
                     "quant_conv", "zq", "post_quant", "dec_mid"]
        names += (0 ... 5).reversed().map { "dec_level\($0)" }

        for name in names {
            guard let got = taps[name] else { fail("missing tap \(name)") }
            let tol: Float = name.hasPrefix("dec_") ? 5e-4 : 1e-4
            r.check("\(tag).\(name)", toNCHW(got), g(dir, "\(tag)_\(name)"), tol: tol)
        }
        r.check("\(tag).image", toNCHW(image), g(dir, "\(tag)_image"), tol: 5e-4)

        let want = loadIndices("\(dir)/\(tag)_indices.npy")
        let got = taps["indices"]!.asArray(Int64.self)
        let mismatches = zip(got, want).filter { $0 != $1 }.count
        print("  \(mismatches == 0 ? "✅" : "❌") \(tag).indices        \(want.count - mismatches)/\(want.count) exact")
        if mismatches != 0 { indicesOK = false }
    }
    return r.summarize() && indicesOK
}

/// e2e at a candidate publish dtype, GPU stream — the dtype-choice gate.
func gateDtype(_ dir: String, _ w: String, dtype: DType, label: String) {
    _ = _unbuffered
    print("=== DTYPE · \(label) e2e (GPU stream) ===\n")
    let model = loadedModel(w)
    let params = model.parameters().mapValues { $0.asType(dtype) }
    model.update(parameters: params)
    eval(model)

    for tag in ["full_rand", "full_face"] {
        let x = toNHWC(g(dir, "\(tag)_in")).asType(dtype)
        let out = model(x)
        eval(out)
        let want = g(dir, "\(tag)_image")
        let got = toNCHW(out.asType(.float32))
        let p = parity(got, want)
        let a = clip((got + 1) / 2, min: 0, max: 1)
        let b = clip((want + 1) / 2, min: 0, max: 1)
        let mse = MLX.mean(MLX.square(a - b)).item(Float.self)
        let psnr = mse > 0 ? 10 * log10(1.0 / mse) : Float.infinity
        print(String(format: "  %@  cos=%.6f  rel=%.3e  PSNR=%.2f dB vs fp32 golden",
                     tag, p.cosine, p.relative, psnr))
    }
}

/// Split footprint on the GPU stream. Face restore is fixed-size 512².
func gateBench(_ w: String) {
    _ = _unbuffered
    print("=== BENCH · split footprint (GPU stream) ===\n")
    let base = physFootprintBytes()
    let model = loadedModel(w)
    MLX.Memory.clearCache()
    let floor = physFootprintBytes()
    print("  post-load floor : \(gb(floor))  → resident ≈ \(gb(floor > base ? floor - base : 0))")
    print("  (weights are 73,472,579 params @ fp32 = 293.9 MB)\n")

    MLX.Memory.clearCache()
    MLX.Memory.peakMemory = 0
    let x = MLXArray.zeros([1, 512, 512, 3], dtype: .float32)
    let t0 = Date()
    let out = model(x)
    eval(out)
    let dt = Date().timeIntervalSince(t0)
    let mlxPeak = MLX.Memory.peakMemory
    let phys = physFootprintBytes()
    print(String(format: "  512x512: MLX peak %@   phys %@   activation ≈ %@   %.2fs",
                 gb(mlxPeak), gb(phys), gb(phys > floor ? phys - floor : 0), dt))
    MLX.Memory.clearCache()
}

func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(max(0, b)) / 1e9) }
func gb(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / 1e9) }

let args = Array(CommandLine.arguments.dropFirst())
guard let mode = args.first else {
    print("usage: restoreformer-gate --s0 <weights> | --s1|--s3|--all <goldens> <weights> "
        + "| --fp16|--bf16 <goldens> <weights> | --bench <weights>")
    exit(2)
}
switch mode {
case "--bench":
    guard args.count >= 2 else { fail("--bench needs a weights path") }
    Device.setDefault(device: .gpu)
    gateBench(args[1])
case "--fp16", "--bf16":
    guard args.count >= 3 else { fail("\(mode) needs <goldens> <weights>") }
    Device.setDefault(device: .gpu)
    gateDtype(args[1], args[2], dtype: mode == "--fp16" ? .float16 : .bfloat16,
              label: mode == "--fp16" ? "fp16" : "bf16")
case "--s0":
    Device.setDefault(device: .cpu)
    guard args.count >= 2 else { fail("--s0 needs a weights path") }
    gateS0(args[1])
case "--s1", "--s3", "--all":
    Device.setDefault(device: .cpu)
    guard args.count >= 3 else { fail("\(mode) needs <goldens> <weights>") }
    let (dir, w) = (args[1], args[2])
    var ok = true
    if mode == "--s1" || mode == "--all" { ok = gateS1(dir, w) && ok; print("") }
    if mode == "--s3" || mode == "--all" { ok = gateS3(dir, w) && ok }
    if !ok { exit(1) }
default: fail("unknown mode \(mode)")
}
