import Foundation
import Hub
import MLXToolKit

/// Where `load()` finds the weights file of the declared `WeightSource` (AB-A-0112).
///
/// The engine materializes `weightSources` into its model store before `load()` (contract 1.24). Before 0.2.1, `load()`
/// didn't look there: it fetched through swift-transformers' `HubApi(downloadBase:)`, which resolves a different
/// layout, so every first engine run downloaded the checkpoint twice. The file is now found in one of three layouts,
/// tried in order:
/// - the engine's flat layout, `<root>/models--<org>--<name>/<file>`;
/// - the hub-client snapshot, `<root>/models--<org>--<name>/snapshots/<commit>/<file>`;
/// - swift-transformers' `HubApi(downloadBase: root)` layout, `<root>/models/<org>/<name>/<file>`, where `load()`
///   downloaded before 0.2.1. Existing installs keep loading from there, with no second download.
///
/// `HubApi` stays only as the backstop for a caller outside the engine (no store, or nothing materialized).
enum StoreWeights {
    static let weightsFile = "model.safetensors"

    /// `load()`'s weights file: the store's copy, else `download`, by default a `HubApi` snapshot into the pre-0.2.1
    /// layout, which `missingWeightSources` also accepts. The tests pass a `download` that fails, to prove a stored
    /// copy loads with no network call.
    static func resolve(repo: String, root: URL?,
                        download: (_ repo: String, _ root: URL?) async throws -> URL = hubSnapshot) async throws -> URL {
        if let stored = file(weightsFile, repo: repo, root: root) { return stored }
        return try await download(repo, root)
    }

    static func file(_ name: String, repo: String, root: URL?) -> URL? {
        guard let root else { return nil }
        let store = ModelStore(root: root)
        let directories = [store.directory(for: repo), store.snapshotDirectory(for: repo), hubAPIDirectory(repo, root: root)]
        return directories.lazy.compactMap { $0?.appendingPathComponent(name) }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// The pre-0.2.1 download location.
    static func hubAPIDirectory(_ repo: String, root: URL) -> URL {
        root.appendingPathComponent("models").appendingPathComponent(repo)
    }

    /// Whether `name` is installed in the pre-0.2.1 layout, which the engine's own probe doesn't know.
    static func installedByHubAPI(_ name: String, repo: String, root: URL?) -> Bool {
        guard let root else { return false }
        return FileManager.default.fileExists(atPath: hubAPIDirectory(repo, root: root).appendingPathComponent(name).path)
    }

    /// The backstop outside the engine: a swift-transformers snapshot, with byte progress.
    static func hubSnapshot(repo: String, root: URL?) async throws -> URL {
        let hub = root.map { HubApi(downloadBase: $0) } ?? HubApi()
        let dir = try await hub.snapshot(from: Hub.Repo(id: repo), matching: [weightsFile]) { progress, speed in
            WeightDownloadProgress.report(fraction: progress.fractionCompleted, bytesPerSecond: speed)
        }
        return dir.appendingPathComponent(weightsFile)
    }
}
