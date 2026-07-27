// ConformanceTests.swift — RestoreFormer++ through the engine's offline gates (no MLX kernels run).
//
// The package-specific angle: this is the first imageRestore backer whose run() is a
// multi-stage pipeline around the forward (Vision detect → align → restore per face →
// paste back), so the CAN cadence unit is a *face*, and the no-face path is a typed no-op
// (input passes through, appliedStrength nil — contract 1.30.0's "no dial here" signal).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
import RestoreFormerMLXCore
@testable import MLXRestoreFormer

final class ConformanceTests: XCTestCase {

    // MARK: - MAT

    func testMATGate() {
        let report = MaterializationConformance.check(freshConfiguration: RestoreFormerConfiguration())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesDeclared() {
        let sources = RestoreFormerConfiguration().weightSources
        XCTAssertEqual(sources.count, 1)
        XCTAssertEqual(sources[0].repo, "mlx-community/RestoreFormerPlusPlus-fp32")
        XCTAssertEqual(sources[0].matching, ["model.safetensors"])
    }

    func testExplicitWeightsURLSuppressesMaterialization() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("restoreformer-\(UUID().uuidString).safetensors")
        try Data([0x00]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        XCTAssertTrue(RestoreFormerConfiguration(weightsURL: tmp).missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(
            RestoreFormerConfiguration(weightsURL: tmp.appendingPathExtension("nope"))
                .missingWeightSources(storeRoot: nil).count, 1)
    }

    // MARK: - CAN

    func testCANGatePreCancelledRun() async {
        let package = RestoreFormerRestorePackage(configuration: RestoreFormerConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageRestoreRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        let manifest = RestoreFormerRestorePackage.manifest
        XCTAssertTrue(CancellationConformance.longRunImplied(by: manifest),
                      "2.0 GB declared activation + multi-face multi-second runs imply long runs")
        // run() checkpoints once per detected face (the real iterative seam) and reports
        // RunProgress on the same unit; the final pre-encode boundary is a frame seam.
        let report = CancellationConformance.checkCadence(
            manifest: manifest,
            posture: .cadence([
                .init(phase: .postprocess, unit: .chunk, reportsRunProgress: true),
                .init(phase: .encode, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Manifest

    func testManifestSurfacesAndLicence() {
        let m = RestoreFormerRestorePackage.manifest
        XCTAssertEqual(m.capabilities, [.imageRestore], "no new capability — same request shape")
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].name, "restoreformer-face-restore")
        XCTAssertEqual(m.license.weightLicense, .apache2)
        XCTAssertEqual(m.license.portCodeLicense, .apache2)
        XCTAssertEqual(m.provenance.sourceRepo, "wzhouxiff/RestoreFormerPlusPlus")
    }

    /// The strength dial is declared on the surface — this backer genuinely honours it
    /// (restored/original crop blend), unlike the frame-restoration siblings.
    func testStrengthParameterIsDeclared() {
        let params = RestoreFormerRestorePackage.manifest.surfaces[0].parameters.map(\.name)
        XCTAssertTrue(params.contains("strength"), "supportsStrength should surface the dial")
    }

    func testFootprintIsSplitAndFixedSize() {
        guard let fp = RestoreFormerRestorePackage.manifest.requirements.footprints
            .first(where: { $0.quant == .fp32 }) else { return XCTFail("no fp32 footprint") }
        // 73.5 M params @ fp32 = 293.9 MB; the floor must cover it without absorbing the
        // activation.
        XCTAssertGreaterThan(fp.residentBytes, 293_900_000)
        XCTAssertLessThan(fp.residentBytes, 800_000_000)
        XCTAssertGreaterThan(fp.peakActivationBytes, fp.residentBytes)
        // Every face runs at exactly 512², so the declared peak must stay one-crop-sized —
        // a declaration far above the measured ~4 GB would mean the fixed-size property broke.
        XCTAssertLessThan(fp.peakActivationBytes, 8_000_000_000)
    }

    func testQuantConfiguredMatchesADeclaredFootprint() {
        let declared = Set(RestoreFormerRestorePackage.manifest.requirements.footprints.map(\.quant))
        for v in RestoreFormerVariant.allCases {
            XCTAssertTrue(declared.contains(RestoreFormerConfiguration(variant: v).quant), "\(v)")
        }
    }

    func testCodableRoundTrip() throws {
        let cfg = RestoreFormerConfiguration(pasteFeather: 12,
                                             modelsRootDirectory: URL(fileURLWithPath: "/x"))
        let decoded = try JSONDecoder().decode(RestoreFormerConfiguration.self,
                                               from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.variant, .plusPlus)
        XCTAssertEqual(decoded.pasteFeather, 12)
        XCTAssertNil(decoded.modelsRootDirectory)   // environment-specific, never encoded
    }

    // MARK: - Alignment math (pure, no Vision)

    /// The similarity solver must reproduce a known transform exactly from 5 points.
    func testSimilarityTransformRecoversKnownTransform() {
        let angle: CGFloat = 0.3, scale: CGFloat = 1.7, tx: CGFloat = 40, ty: CGFloat = -25
        let known = CGAffineTransform(a: scale * cos(angle), b: scale * sin(angle),
                                      c: -scale * sin(angle), d: scale * cos(angle),
                                      tx: tx, ty: ty)
        let src = FaceAlign.template512
        let dst = src.map { $0.applying(known) }
        let recovered = FaceAlign.similarityTransform(from: src, to: dst)
        for (p, q) in zip(src.map({ $0.applying(recovered) }), dst) {
            XCTAssertEqual(p.x, q.x, accuracy: 1e-6)
            XCTAssertEqual(p.y, q.y, accuracy: 1e-6)
        }
    }

    /// Feather mask: 1 in the interior, 0 at the border, monotone ramp between.
    func testFeatherMaskShape() {
        let n = FaceAlign.cropSize
        let mask = FaceAlign.featherMask(feather: 26)
        XCTAssertEqual(mask.count, n * n)
        XCTAssertEqual(mask[(n / 2) * n + n / 2], 1)
        XCTAssertEqual(mask[0], 0)
        XCTAssertEqual(mask[(n / 2) * n], 0)             // border center
        XCTAssertGreaterThan(mask[(n / 2) * n + 13], 0)  // mid-ramp
        XCTAssertLessThan(mask[(n / 2) * n + 13], 1)
    }
}
