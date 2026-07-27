//
//  Goldens.swift
//  mlx-restoreformer-swift / RestoreFormerGate
//
//  Minimal `.npy` reader + parity comparison helpers for the CPU-stream gates.
//
//  The goldens are written by `oracle/gen_goldens.py` as fp32, C-contiguous, PyTorch **NCHW**.
//  This file only handles that exact case deliberately — a permissive reader would silently accept
//  a Fortran-ordered or float64 fixture and compare garbage.
//

import Foundation
import MLX

enum GoldenError: Error, CustomStringConvertible {
    case unreadable(String)
    case badMagic(String)
    case unsupported(String, String)

    var description: String {
        switch self {
        case .unreadable(let p): return "cannot read \(p)"
        case .badMagic(let p): return "\(p) is not a .npy file"
        case .unsupported(let p, let why): return "\(p): \(why)"
        }
    }
}

/// Loads a fp32 C-ordered `.npy` into an MLXArray with its original shape.
func loadNPY(_ path: String) throws -> MLXArray {
    guard let data = FileManager.default.contents(atPath: path) else {
        throw GoldenError.unreadable(path)
    }
    let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]   // \x93NUMPY
    guard data.count > 10, Array(data.prefix(6)) == magic else {
        throw GoldenError.badMagic(path)
    }

    let major = data[6]
    let headerLen: Int
    let headerStart: Int
    if major == 1 {
        headerLen = Int(data[8]) | (Int(data[9]) << 8)
        headerStart = 10
    } else {
        headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
        headerStart = 12
    }
    guard let header = String(data: data[headerStart ..< headerStart + headerLen], encoding: .ascii) else {
        throw GoldenError.unsupported(path, "non-ASCII header")
    }

    guard header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\"") else {
        throw GoldenError.unsupported(path, "expected little-endian float32 ('<f4'); header: \(header)")
    }
    guard header.contains("'fortran_order': False") else {
        throw GoldenError.unsupported(path, "expected C order")
    }

    // shape: (a, b, c)  — also handles the 1-D trailing-comma form (n,)
    guard let open = header.firstIndex(of: "("), let close = header[open...].firstIndex(of: ")") else {
        throw GoldenError.unsupported(path, "no shape tuple in header")
    }
    let shape = header[header.index(after: open) ..< close]
        .split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

    let body = data[(headerStart + headerLen)...]
    let count = shape.reduce(1, *)
    guard body.count >= count * 4 else {
        throw GoldenError.unsupported(path, "truncated: need \(count * 4) bytes, have \(body.count)")
    }
    let floats: [Float] = body.withUnsafeBytes { raw in
        Array(UnsafeBufferPointer(start: raw.bindMemory(to: Float.self).baseAddress!, count: count))
    }
    return MLXArray(floats, shape)
}

// MARK: - Layout helpers

/// PyTorch NCHW → MLX NHWC.
func toNHWC(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 1) }

/// MLX NHWC → PyTorch NCHW (for comparing against a golden in its native layout).
func toNCHW(_ x: MLXArray) -> MLXArray { x.transposed(0, 3, 1, 2) }

/// PyTorch patch layout `(B, C, h, w, p1, p2)` → MLX patch layout `(B, h, w, p1, p2, C)`.
func patchToNHWC(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 4, 5, 1) }

/// MLX patch layout → PyTorch patch layout.
func patchToNCHW(_ x: MLXArray) -> MLXArray { x.transposed(0, 5, 1, 2, 3, 4) }

// MARK: - Comparison

struct Parity {
    let maxAbs: Float
    /// `maxAbs / max(|reference|)` — the number the gate actually judges on.
    ///
    /// An absolute threshold is wrong here: these tensors span three orders of magnitude between
    /// ops (a LayerNorm output sits near ±2, a Fuse output on seeded inputs reaches ±2400), so one
    /// absolute tolerance either fails clean fp32 rounding on the large tensors or waves through
    /// real errors on the small ones. Relative error is scale-invariant and comparable across ops.
    let relative: Float
    let cosine: Float
    let refRange: (Float, Float)

    var line: String {
        String(format: "rel=%.2e  max_abs=%.3e  cos=%.8f  (ref %+.1f…%+.1f)",
               relative, maxAbs, cosine, refRange.0, refRange.1)
    }
}

func parity(_ got: MLXArray, _ want: MLXArray) -> Parity {
    let a = got.asType(.float32).flattened()
    let b = want.asType(.float32).flattened()
    let maxAbs = MLX.max(MLX.abs(a - b)).item(Float.self)
    let scale = MLX.max(MLX.abs(b)).item(Float.self)
    let dot = MLX.sum(a * b).item(Float.self)
    let na = MLX.sqrt(MLX.sum(a * a)).item(Float.self)
    let nb = MLX.sqrt(MLX.sum(b * b)).item(Float.self)
    let cos = (na > 0 && nb > 0) ? dot / (na * nb) : (na == nb ? 1 : 0)
    return Parity(maxAbs: maxAbs,
                  relative: scale > 0 ? maxAbs / scale : maxAbs,
                  cosine: cos,
                  refRange: (MLX.min(b).item(Float.self), MLX.max(b).item(Float.self)))
}

/// Records pass/fail across a gate so one run reports every op rather than dying on the first.
final class GateReport {
    private var rows: [(String, Parity, Float, Bool)] = []
    private let name: String
    init(_ name: String) { self.name = name }

    /// - Parameter tol: maximum permitted **relative** error (`maxAbs / max|ref|`).
    func check(_ label: String, _ got: MLXArray, _ want: MLXArray, tol: Float) {
        let p = parity(got, want)
        let ok = p.relative <= tol && !p.relative.isNaN
        rows.append((label, p, tol, ok))
        print("  \(ok ? "✅" : "❌") \(label.padding(toLength: 22, withPad: " ", startingAt: 0)) "
            + "\(p.line)  tol=\(tol)")
    }

    /// - Returns: true if every check passed.
    @discardableResult
    func summarize() -> Bool {
        let failed = rows.filter { !$0.3 }
        print("")
        if failed.isEmpty {
            print("✅ \(name) PASSED — \(rows.count)/\(rows.count) checks within tolerance.")
            return true
        }
        print("❌ \(name) FAILED — \(failed.count)/\(rows.count) checks out of tolerance:")
        for (l, p, tol, _) in failed { print("     \(l): \(p.line) tol=\(tol)") }
        return false
    }
}
