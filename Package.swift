// swift-tools-version: 6.2
import PackageDescription

// mlx-restoreformer-swift — RestoreFormer++ blind face restoration for MLXEngine.
// VQ-GAN encoder/decoder over the ROHQD codebook + multi-scale multi-head cross-attention;
// 512² aligned crops. Upstream: wzhouxiff/RestoreFormerPlusPlus — plain Apache-2.0.
let package = Package(
    name: "mlx-restoreformer-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "RestoreFormerMLXCore", targets: ["RestoreFormerMLXCore"]),
        .library(name: "MLXRestoreFormer", targets: ["MLXRestoreFormer"]),
        .executable(name: "restoreformer-gate", targets: ["RestoreFormerGate"]),
        .executable(name: "restoreformer-validate", targets: ["RestoreFormerValidate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.38.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "RestoreFormerMLXCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MLXRestoreFormer",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "RestoreFormerMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXRestoreFormerTests",
            dependencies: [
                "RestoreFormerMLXCore", "MLXRestoreFormer",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
        .executableTarget(
            name: "RestoreFormerValidate",
            dependencies: [
                "MLXRestoreFormer",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXEngineTestKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/Validate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "RestoreFormerGate",
            dependencies: [
                "RestoreFormerMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/Gate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
