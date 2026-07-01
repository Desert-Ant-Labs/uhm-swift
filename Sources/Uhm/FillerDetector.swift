import CoreML
import Foundation

/// Frame-level filler detector. Wraps a Core ML model that emits a
/// per-frame softmax (20 ms frames) over filler classes; this class
/// thresholds and merges consecutive positive frames into (start, end)
/// detections. Pair with `FillerTypeClassifier` to assign a `FillerType`.
final class FillerDetector: @unchecked Sendable {

    // MARK: - Configuration

    struct Config: Sendable {
        var sampleRate: Double
        var maxWindowSec: Double      // model's fixed input length (e.g. 30s)
        var hopSec: Double            // sliding window hop between successive model calls
        var frameHopSamples: Int      // conv stem hop (320 = 20ms @ 16kHz)
        var minFrameProb: Double      // per-frame threshold for "is filler"
        var minDurationSec: Double    // discard runs shorter than this
        var mergeGapSec: Double       // merge adjacent runs within this gap

        static let `default` = Config(
            sampleRate: 16_000,
            maxWindowSec: 30.0,
            hopSec: 25.0,                // 5s overlap between windows
            frameHopSamples: 320,
            minFrameProb: 0.5,
            minDurationSec: 0.10,
            mergeGapSec: 0.10
        )
    }

    /// Bucketed wall-time captured during a single `detect()` call. Exposed via
    /// the `timingsHandler` for diagnostic / bench callers that want to see
    /// where inference time is going.
    struct Timings: Sendable {
        /// Total `model.predictions(from:)` wall time across every batch.
        /// Everything ANE/GPU-side lives here.
        var inferenceSec: Double
        /// Per-window normalize + MLFeatureProvider build.
        var prepSec: Double
        /// Frame-prob threshold + run merging.
        var groupSec: Double

        init(inferenceSec: Double = 0, prepSec: Double = 0, groupSec: Double = 0) {
            self.inferenceSec = inferenceSec
            self.prepSec = prepSec
            self.groupSec = groupSec
        }
    }

    let config: Config
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let maxSamples: Int

    // MARK: - Init

    /// Loads the model, letting Core ML choose compute units.
    convenience init(modelPath: String, config: Config = .default) throws {
        try self.init(modelPath: modelPath, config: config, configuration: nil)
    }

    /// Loads the model with an explicit `MLModelConfiguration` so callers can
    /// pin `computeUnits` (diagnostic A/B of ANE vs GPU vs CPU). Pass `nil` to
    /// keep Core ML's default behavior.
    init(modelPath: String, config: Config = .default, configuration: MLModelConfiguration?) throws {
        var c = config
        if let s = ProcessInfo.processInfo.environment["UHM_FRAME_MIN_PROB"], let v = Double(s) {
            c.minFrameProb = v
        }

        let url = URL(fileURLWithPath: modelPath)
        let compiled = url.pathExtension == "mlmodelc" ? url : try Self.compiledURL(for: url)
        let model: MLModel
        if let configuration {
            model = try MLModel(contentsOf: compiled, configuration: configuration)
        } else {
            model = try MLModel(contentsOf: compiled)
        }
        self.model = model
        self.inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "audio"
        self.outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "filler_probs"
        // Prefer the model's actual input sample count (read from
        // user-defined metadata at convert time) over the config default.
        // Lets bench-only variants ship with shorter / longer windows
        // (e.g. a 15 s window variant) without forking the detector.
        // Keep hop sec relative to window — preserve 5 s overlap unless
        // the caller has explicitly set a non-default hop.
        let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        if let s = meta["max_samples"], let n = Int(s), n > 0 {
            self.maxSamples = n
            c.maxWindowSec = Double(n) / c.sampleRate
            if config.hopSec == Config.default.hopSec {
                c.hopSec = max(1.0, c.maxWindowSec - 5.0)
            }
        } else {
            self.maxSamples = Int(c.maxWindowSec * c.sampleRate)
        }
        self.config = c
    }

    /// Persist the compiled `.mlmodelc` so subsequent inits skip the
    /// ~1–2 s recompile. The default download path already provides a
    /// precompiled `.mlmodelc` (returned as-is); this only compiles when a
    /// caller points `Uhm(modelURL:)` at a local `.mlpackage`. Prefers a
    /// sibling location next to the source; falls back to
    /// `Caches/uhm-compiled/` when the source lives in a read-only
    /// directory (app bundle for bench-bundled variants).
    private static func compiledURL(for source: URL) throws -> URL {
        let stem = source.deletingPathExtension().lastPathComponent
        let sourceDir = source.deletingLastPathComponent()
        let preferred = sourceDir.appendingPathComponent("\(stem).mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: preferred.path) {
            return preferred
        }
        // Pick a cache target: sibling-to-source if writable, else
        // Caches/uhm-compiled/<stem>.mlmodelc. The latter survives across
        // app launches but gets purged on iOS low-storage events — same
        // semantics as URLCache, fine for a recompile-on-demand cache.
        let isWritable = FileManager.default.isWritableFile(atPath: sourceDir.path)
        let cached: URL
        if isWritable {
            cached = preferred
        } else {
            let caches = try FileManager.default.url(
                for: .cachesDirectory, in: .userDomainMask,
                appropriateFor: nil, create: true)
            let dir = caches.appendingPathComponent("uhm-compiled", isDirectory: true)
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            cached = dir.appendingPathComponent("\(stem).mlmodelc", isDirectory: true)
            if FileManager.default.fileExists(atPath: cached.path) {
                return cached
            }
        }
        let compiled = try MLModel.compileModel(at: source)
        let tmp = cached.appendingPathExtension("tmp.\(UUID().uuidString)")
        try? FileManager.default.removeItem(at: tmp)
        try FileManager.default.copyItem(at: compiled, to: tmp)
        if FileManager.default.fileExists(atPath: cached.path) {
            try? FileManager.default.removeItem(at: cached)
        }
        try FileManager.default.moveItem(at: tmp, to: cached)
        return cached
    }

    // MARK: - Detection

    /// Detect filler spans in decoded mono samples (at `config.sampleRate`).
    ///
    /// Passing an optional `timingsHandler` opts into a per-phase wall-time
    /// breakdown; the timer overhead is negligible and the handler runs
    /// synchronously before returning.
    func detect(
        samples: [Float],
        progressHandler: ((Double) -> Void)? = nil,
        timingsHandler: ((Timings) -> Void)? = nil
    ) throws -> [Filler] {
        var t = Timings()
        let measure = timingsHandler != nil
        // Early bail if the caller already cancelled before we started.
        try Task.checkCancellation()
        guard !samples.isEmpty else {
            timingsHandler?(t)
            return []
        }

        // Run sliding windows of `maxWindowSec` with `hopSec` hop, average overlapping frame probs.
        let stepSamples = Int(config.hopSec * config.sampleRate)
        let totalFrames = (samples.count + config.frameHopSamples - 1) / config.frameHopSamples
        var sumProbs = [Float](repeating: 0, count: totalFrames)
        var counts = [Float](repeating: 0, count: totalFrames)

        // Collect window offsets up-front, then run them in batches so Core ML
        // can pipeline across compute units instead of one-at-a-time submits.
        var winOffsets: [(start: Int, end: Int)] = []
        var winStart = 0
        while winStart < samples.count {
            let end = min(samples.count, winStart + maxSamples)
            winOffsets.append((winStart, end))
            if end == samples.count { break }
            winStart += stepSamples
        }

        let batchSize = max(1, Int(ProcessInfo.processInfo.environment["UHM_BATCH_SIZE"] ?? "") ?? 4)
        progressHandler?(0)
        var processed = 0
        for chunkStart in stride(from: 0, to: winOffsets.count, by: batchSize) {
            // Honour `Task.cancel()` between batches — gives callers a clean
            // exit point in the middle of long files. Throws CancellationError
            // which bubbles up through `detect()` and `Uhm.analyze()`.
            try Task.checkCancellation()
            let chunk = winOffsets[chunkStart..<min(chunkStart + batchSize, winOffsets.count)]
            let tPrep = measure ? CFAbsoluteTimeGetCurrent() : 0
            let providers: [MLFeatureProvider] = try chunk.map { range in
                try makeProvider(window: samples, start: range.start, end: range.end)
            }
            let batch = MLArrayBatchProvider(array: providers)
            if measure { t.prepSec += CFAbsoluteTimeGetCurrent() - tPrep }
            let tInf = measure ? CFAbsoluteTimeGetCurrent() : 0
            let outBatch = try model.predictions(from: batch, options: MLPredictionOptions())
            if measure { t.inferenceSec += CFAbsoluteTimeGetCurrent() - tInf }
            for (i, range) in chunk.enumerated() {
                let probs = decodeProbs(from: outBatch.features(at: i))
                let frameOffset = range.start / config.frameHopSamples
                let usableFrames = (range.end - range.start + config.frameHopSamples - 1) / config.frameHopSamples
                for k in 0..<min(usableFrames, probs.count) {
                    let g = frameOffset + k
                    if g < totalFrames {
                        sumProbs[g] += probs[k]
                        counts[g] += 1
                    }
                }
            }
            processed += chunk.count
            progressHandler?(min(1.0, Double(processed) / Double(winOffsets.count)))
        }

        // Average overlapping windows
        var probs = [Float](repeating: 0, count: totalFrames)
        for i in 0..<totalFrames {
            probs[i] = counts[i] > 0 ? sumProbs[i] / counts[i] : 0
        }

        // Threshold + merge runs
        let tGroup = measure ? CFAbsoluteTimeGetCurrent() : 0
        let threshold = Float(config.minFrameProb)
        let frameSec = Double(config.frameHopSamples) / config.sampleRate
        var fillers: [Filler] = []
        var i = 0
        while i < probs.count {
            if probs[i] < threshold { i += 1; continue }
            var j = i
            var sum: Float = 0
            while j < probs.count && probs[j] >= threshold {
                sum += probs[j]
                j += 1
            }
            // Merge with previous if gap is small enough
            let startSec = Double(i) * frameSec
            let endSec = Double(j) * frameSec
            let avgConf = Double(sum / Float(j - i))
            if let last = fillers.last, startSec - last.end <= config.mergeGapSec {
                fillers[fillers.count - 1] = Filler(
                    label: "filler",
                    start: last.start, end: endSec,
                    confidence: max(last.confidence, avgConf)
                )
            } else if endSec - startSec >= config.minDurationSec {
                fillers.append(Filler(
                    label: "filler",
                    start: startSec, end: endSec,
                    confidence: avgConf
                ))
            }
            i = j
        }
        if measure { t.groupSec = CFAbsoluteTimeGetCurrent() - tGroup }
        timingsHandler?(t)
        return fillers
    }

    // MARK: - Core ML helpers

    /// Build a single-window MLFeatureProvider against the source sample
    /// buffer. Splitting prep out lets us collect many providers and submit
    /// them as one batch via `model.predictions(from:)`.
    private func makeProvider(window samples: [Float], start: Int, end: Int) throws -> MLFeatureProvider {
        // Per-window mean/std normalize, matching the feature extractor used in training.
        var mean: Float = 0
        for i in start..<end { mean += samples[i] }
        mean /= Float(end - start)
        var sumSq: Float = 0
        for i in start..<end { sumSq += (samples[i] - mean) * (samples[i] - mean) }
        let std = (sumSq / Float(end - start - 1)).squareRoot() + 1e-7
        let invStd = 1 / std

        let arr = try MLMultiArray(shape: [1, maxSamples] as [NSNumber], dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        let n = end - start
        for k in 0..<n { ptr[k] = (samples[start + k] - mean) * invStd }
        if n < maxSamples {
            for k in n..<maxSamples { ptr[k] = 0 }
        }
        return try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
    }

    private func decodeProbs(from out: MLFeatureProvider) -> [Float] {
        guard let probs = out.featureValue(for: outputName)?.multiArrayValue else { return [] }
        // Multiclass softmax output: shape (1, T, C). Per-frame filler probability
        // is 1 - p_not_filler (class 0). Use the array's reported strides — Core ML
        // pads inner dims to 16/32-byte boundaries for ANE alignment, and uses
        // Float16 storage on Apple Silicon, so naive Float32 reads return garbage.
        let shape = probs.shape.map { $0.intValue }
        let strides = probs.strides.map { $0.intValue }
        if shape.count >= 3 {
            let T = shape[shape.count - 2]
            let tStride = strides[shape.count - 2]
            let cStride = strides[shape.count - 1]
            var result = [Float](); result.reserveCapacity(T)
            // assumingMemoryBound (vs bindMemory) — strided storage is larger
            // than logical `probs.count`; bindMemory's capacity check segfaults.
            switch probs.dataType {
            case .float16:
                let ptr = probs.dataPointer.assumingMemoryBound(to: Float16.self)
                for t in 0..<T {
                    let pNot = Float(ptr[t * tStride + 0 * cStride])
                    result.append(1.0 - pNot)
                }
            case .float32:
                let ptr = probs.dataPointer.assumingMemoryBound(to: Float.self)
                for t in 0..<T {
                    let pNot = ptr[t * tStride + 0 * cStride]
                    result.append(1.0 - pNot)
                }
            case .float64:
                let ptr = probs.dataPointer.assumingMemoryBound(to: Double.self)
                for t in 0..<T {
                    let pNot = Float(ptr[t * tStride + 0 * cStride])
                    result.append(1.0 - pNot)
                }
            default:
                for t in 0..<T {
                    let pNot = Float(truncating: probs[[0, NSNumber(value: t), 0]])
                    result.append(1.0 - pNot)
                }
            }
            return result
        }
        var result = [Float](); result.reserveCapacity(probs.count)
        for k in 0..<probs.count {
            result.append(Float(truncating: probs[k]))
        }
        return result
    }
}
