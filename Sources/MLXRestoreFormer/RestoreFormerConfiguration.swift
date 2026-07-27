import Foundation
import MLXToolKit

/// A RestoreFormer checkpoint. `plusPlus` (TPAMI 2023) is the primary; the original
/// RestoreFormer (CVPR 2022) shares the loader shape but a different architecture config
/// (ch=128, single-scale attention) — add it only if a corpus shows it earns a slot.
public enum RestoreFormerVariant: String, Codable, Sendable, CaseIterable {
    /// RestoreFormer++ — multi-scale cross-attention, the release default.
    case plusPlus

    public var repo: String {
        switch self {
        case .plusPlus: return "mlx-community/RestoreFormerPlusPlus-fp32"
        }
    }

    /// 294 MB at fp32. Measured dtype gates (vs the fp32 golden, face fixture): fp16
    /// 50.1 dB (viable — mantissa-bound, it BEATS bf16), bf16 38.6 dB. The late-decoder
    /// activations reach ±14k, close enough to fp16's 65504 ceiling that fp32 stays the
    /// safe ship per the restoration-family precedent. Measure before changing.
    public var quant: Quant { .fp32 }
}

/// Init-time configuration for `RestoreFormerRestorePackage` (C9).
public struct RestoreFormerConfiguration: PackageConfiguration, ModelStorable {
    public var variant: RestoreFormerVariant

    /// Border feather (px, in the 512² crop) for the paste-back mask.
    public var pasteFeather: Int

    /// Faces whose eye distance is below this (px) are skipped — upstream's
    /// `eye_dist_threshold=5` guard against spurious detections.
    public var minEyeDistance: Float

    public var modelsRootDirectory: URL?
    public var weightsURL: URL?

    public init(variant: RestoreFormerVariant = .plusPlus,
                pasteFeather: Int = 26,
                minEyeDistance: Float = 5,
                modelsRootDirectory: URL? = nil,
                weightsURL: URL? = nil) {
        self.variant = variant
        self.pasteFeather = pasteFeather
        self.minEyeDistance = minEyeDistance
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsURL = weightsURL
    }

    private enum CodingKeys: String, CodingKey {
        case variant, pasteFeather, minEyeDistance
    }
}

extension RestoreFormerConfiguration: QuantConfigured {
    public var quant: Quant { variant.quant }
}

extension RestoreFormerConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "weights", repo: variant.repo, revision: nil,
                      matching: ["model.safetensors"])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let weightsURL, FileManager.default.fileExists(atPath: weightsURL.path) { return [] }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}
