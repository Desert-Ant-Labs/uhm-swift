// swift-tools-version: 5.10
import PackageDescription

// uhm-swift. On-device filler-word detection for Apple platforms.
//
// The detector downloads the single shipped Core ML model (~45 MB) from
// huggingface.co/desert-ant-labs/uhm on first use via DesertAntStore's Hugging
// Face Hub snapshot and caches it locally. The tiny type-labeler bundles as a
// package resource.
let package = Package(
    name: "Uhm",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "Uhm", targets: ["Uhm"]),
    ],
    dependencies: [
        .package(url: "https://github.com/Desert-Ant-Labs/desert-ant-swift.git", revision: "026274847205b923a9e6dcbf8db52f95a734c2cb"),
    ],
    targets: [
        .target(
            name: "Uhm",
            dependencies: [
                .product(name: "DesertAntStore", package: "desert-ant-swift"),
            ],
            path: "Sources/Uhm",
            resources: [
                .process("Resources"),
            ]
        ),
        .testTarget(
            name: "UhmTests",
            dependencies: ["Uhm"],
            path: "Tests/UhmTests"
        ),
    ]
)
