// StoreWeightsTests.swift — load() reads the copy the engine materialized, so an engine run downloads the checkpoint
// once, into one layout (AB-A-0112). Offline: no network, no weights, no MLX kernels.

import Foundation
import MLXToolKit
import XCTest
@testable import MLXRestoreFormer

final class StoreWeightsTests: XCTestCase {
    private var root: URL!
    private var configuration: RestoreFormerConfiguration { RestoreFormerConfiguration(modelsRootDirectory: root) }
    private var repo: String { configuration.weightSources[0].repo }

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appending(path: "restoreformer-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    @discardableResult
    private static func place(_ directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appending(path: StoreWeights.weightsFile)
        try Data([0]).write(to: file)
        return file
    }

    /// The `download` for a store that already holds the file: reaching it is the bug.
    private static func noNetwork(_ repo: String, _ root: URL?) async throws -> URL {
        XCTFail("load() went to the network for \(repo) though the store holds it")
        throw URLError(.notConnectedToInternet)
    }

    /// Every weights file under the store, in any layout.
    private func copiesOnDisk() -> Int {
        let all = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? []
        return all.filter { $0.lastPathComponent == StoreWeights.weightsFile }.count
    }

    /// An empty store: the probe reports the source, so the engine downloads it into its flat layout (placed here, as
    /// `MLXServeEngine`'s materializer does). load() then reads that copy without a download of its own: one copy.
    func testEmptyStoreDownloadsOnce() async throws {
        XCTAssertEqual(configuration.missingWeightSources(storeRoot: root).map(\.repo), [repo])
        let engineCopy = try Self.place(ModelStore(root: root).directory(for: repo)!)
        XCTAssertTrue(configuration.missingWeightSources(storeRoot: root).isEmpty)
        let url = try await StoreWeights.resolve(repo: repo, root: root, download: Self.noNetwork)
        XCTAssertEqual(url.path, engineCopy.path)
        XCTAssertEqual(copiesOnDisk(), 1)
    }

    /// A store holding only the engine's copy loads with no network call, in its flat and its snapshot layout.
    func testEngineLayoutLoadsWithoutNetwork() async throws {
        let repoDirectory = ModelStore(root: root).directory(for: repo)!
        let flat = try Self.place(repoDirectory)
        var url = try await StoreWeights.resolve(repo: repo, root: root, download: Self.noNetwork)
        XCTAssertEqual(url.path, flat.path)

        try FileManager.default.removeItem(at: flat)
        try FileManager.default.createDirectory(at: repoDirectory.appending(path: "refs"), withIntermediateDirectories: true)
        try "abc123".write(to: repoDirectory.appending(path: "refs/main"), atomically: true, encoding: .utf8)
        let snapshot = try Self.place(repoDirectory.appending(path: "snapshots/abc123"))
        XCTAssertTrue(configuration.missingWeightSources(storeRoot: root).isEmpty)
        url = try await StoreWeights.resolve(repo: repo, root: root, download: Self.noNetwork)
        XCTAssertEqual(url.path, snapshot.path)
    }

    /// Before 0.2.1, load() downloaded through `HubApi(downloadBase:)` into `<root>/models/<org>/<name>/`. A store holding
    /// only that copy reads as present, so the engine doesn't fetch it again, and load() uses it with no network call.
    func testOldHubApiInstallIsNotFetchedAgain() async throws {
        let old = try Self.place(root.appending(path: "models").appending(path: repo))
        XCTAssertTrue(configuration.missingWeightSources(storeRoot: root).isEmpty)
        let url = try await StoreWeights.resolve(repo: repo, root: root, download: Self.noNetwork)
        XCTAssertEqual(url.path, old.path)
        XCTAssertEqual(copiesOnDisk(), 1)
    }

    /// Outside the engine nothing is materialized: the backstop downloads once, into a layout the probe accepts, so a
    /// later engine run doesn't fetch it again either.
    func testBackstopDownloadsOnceIntoAnAcceptedLayout() async throws {
        _ = try await StoreWeights.resolve(repo: repo, root: root) { repo, root in
            try Self.place(StoreWeights.hubAPIDirectory(repo, root: root!))
        }
        XCTAssertTrue(configuration.missingWeightSources(storeRoot: root).isEmpty)
        _ = try await StoreWeights.resolve(repo: repo, root: root, download: Self.noNetwork)
        XCTAssertEqual(copiesOnDisk(), 1)
    }

    func testNoStoreFindsNothing() {
        XCTAssertNil(StoreWeights.file(StoreWeights.weightsFile, repo: repo, root: nil))
        XCTAssertEqual(RestoreFormerConfiguration().missingWeightSources(storeRoot: nil).count, 1)
    }
}
