import CoreML
import DesertAntStore
import Foundation

/// `Uhm` — public API for filler-word detection.
///
/// The Core ML model is downloaded on demand from `desert-ant-labs/uhm` and
/// cached locally. Uhm ships one runtime model: the smaller, faster, more
/// precise DistilHuBERT classifier.
///
/// Construction is cheap and does no I/O; the model downloads (if needed) and
/// loads on the first `analyze(...)`. Call `downloadModel(progress:)` to prewarm.
///
/// Usage:
/// ```
/// let uhm = Uhm()
/// let result = try await uhm.analyze(audioPath: "in.wav")
/// for f in result.fillers { print(f.start, f.end, f.type ?? "") }
/// ```
public final class Uhm: @unchecked Sendable {

    // MARK: - Types

    /// Precision / recall trade-off knob. `balanced` is the default — empirically
    /// the cleanest cutoff between real fillers and borderline false positives
    /// across en/es/fr/de/nl. Step up to `.precision` for stricter auto-cut, or
    /// down to `.recall` when you'd rather review-and-confirm than miss.
    public enum Bias: Sendable {
        // Thresholds are stable across model versions: shipped models are
        // pre-calibrated so a given `Options.minConfidence` means the same thing
        // against any release.
        /// Strictest gate (min confidence 0.75) — fewest false alarms; safest for automatic cuts.
        case precision
        /// Default gate (min confidence 0.65) — clean cuts on the labeled corpus.
        case balanced
        /// Loosest gate (min confidence 0.50) — catches more, at the cost of more false positives.
        case recall

        /// The confidence threshold this preset maps to. Public so host apps can
        /// reuse a preset's value directly (e.g. gate an edit action at
        /// `Bias.precision.minConfidence`) instead of hardcoding the number.
        public var minConfidence: Double {
            switch self {
            case .precision: return 0.75
            case .balanced:  return 0.65
            case .recall:    return 0.50
            }
        }
    }

    /// Tuning for a single `analyze(...)` call.
    public struct Options: Sendable {
        /// Precision/recall preset that sets the confidence threshold. Default `.balanced`.
        public var bias: Bias
        /// Whether to run the bundled type labeler to fill in `Detection.type`
        /// (`uh`/`um`/`hmm`/…). Set `false` to skip it when you only need
        /// filler-vs-not spans. Default `true`.
        public var includeTypes: Bool
        /// Override the bias preset's confidence threshold. `nil` = use `bias`.
        public var minConfidence: Double?
        /// Discard detections shorter than this, in seconds. Default `0.12`.
        public var minDurationSec: Double

        /// Creates analysis options.
        /// - Parameters:
        ///   - bias: Precision/recall preset. Default `.balanced`.
        ///   - includeTypes: Run the type labeler. Default `true`.
        ///   - minConfidence: Explicit threshold that overrides `bias`. Default `nil`.
        ///   - minDurationSec: Minimum detection duration to keep. Default `0.12`.
        public init(bias: Bias = .balanced,
                    includeTypes: Bool = true,
                    minConfidence: Double? = nil,
                    minDurationSec: Double = 0.12) {
            self.bias = bias
            self.includeTypes = includeTypes
            self.minConfidence = minConfidence
            self.minDurationSec = minDurationSec
        }

        /// The default options (`.balanced`, types on, 0.12 s minimum).
        public static let `default` = Options()
    }

    /// Output of the bundled per-filler type labeler.
    /// `and` is a mid-sentence "and"-as-filler subtype added in the v3
    /// classifier — useful when you want to keep or treat connectors
    /// differently from `uh`/`um`. `other` is the "labeler isn't confident
    /// which kind" bucket — surface it as something neutral ("filler") in
    /// user-facing UI; useful as-is for analytics.
    public enum FillerType: String, Sendable, Codable, CaseIterable {
        case uh, um, hmm, and, other
    }

    /// A single detected filler span.
    public struct Detection: Sendable {
        /// Start time in seconds from the beginning of the audio.
        public let start: Double
        /// End time in seconds from the beginning of the audio.
        public let end: Double
        /// Model confidence for this span, in `0...1`.
        public let confidence: Double
        /// The filler subtype, or `nil` when `includeTypes` is false or the type labeler couldn't load.
        public let type: FillerType?
        /// Span length in seconds (`end - start`).
        public var duration: Double { end - start }

        /// Creates a detection. Normally you receive these from `analyze(...)`
        /// rather than constructing them, but the initializer is public for
        /// tests and for feeding `reconcileWords(_:fillers:)`.
        public init(start: Double, end: Double, confidence: Double, type: FillerType?) {
            self.start = start
            self.end = end
            self.confidence = confidence
            self.type = type
        }
    }

    /// Core ML compute-unit policy. Defaults to `.all` (the same as Core ML
    /// picks when no `MLModelConfiguration` is supplied); the other cases
    /// are diagnostic — useful for A/B'ing whether ANE actually engages on
    /// this device, or for forcing a specific accelerator while debugging.
    public enum ComputeUnits: Sendable {
        /// CPU + GPU + ANE; Core ML picks per op. Default.
        case all
        /// CPU + Apple Neural Engine only (skip GPU) — useful for ANE A/B.
        case cpuAndNeuralEngine
        /// CPU + GPU only (skip ANE).
        case cpuAndGPU
        /// CPU only — baseline / reference.
        case cpuOnly

        var mlValue: MLComputeUnits {
            switch self {
            case .all:                return .all
            case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
            case .cpuAndGPU:          return .cpuAndGPU
            case .cpuOnly:            return .cpuOnly
            }
        }
    }

    /// Per-phase wall-time captured during `analyze()`. `inferenceSec` is
    /// the sum of every Core ML predict call (the bit that ANE/GPU touch);
    /// the rest is Swift glue. Use it to tell whether you're CPU-bound on
    /// decode/labeling or actually waiting on the model.
    public struct PhaseTimings: Sendable {
        /// `AVAudioFile` → 16 kHz mono Float32. One pass over the audio.
        public var decodeSec: Double
        /// Cumulative `MLModel.predictions(from:)` time inside the frame
        /// detector. This is the number that moves when you swap variants
        /// or pin compute units.
        public var inferenceSec: Double
        /// Per-window normalize + MLFeatureProvider build.
        public var prepSec: Double
        /// Threshold + run merging on frame probs.
        public var groupSec: Double
        /// Bundled type-labeler classifier across all detections (small
        /// model, mostly CPU; usually a few ms total).
        public var labelingSec: Double

        /// Creates a phase-timing record. Populated by `analyze(...)`; the
        /// initializer is public mainly for constructing `Result` values in tests.
        public init(decodeSec: Double = 0,
                    inferenceSec: Double = 0,
                    prepSec: Double = 0,
                    groupSec: Double = 0,
                    labelingSec: Double = 0) {
            self.decodeSec = decodeSec
            self.inferenceSec = inferenceSec
            self.prepSec = prepSec
            self.groupSec = groupSec
            self.labelingSec = labelingSec
        }
    }

    /// The result of an `analyze(...)` call.
    public struct Result: Sendable {
        /// Detected fillers, in time order.
        public let fillers: [Detection]
        /// Total decoded audio length, in seconds.
        public let audioDuration: Double
        /// Per-phase breakdown of the `analyze(...)` call this `Result` came from.
        public let phaseTimings: PhaseTimings

        /// Creates a result. You normally receive this from `analyze(...)`.
        public init(fillers: [Detection], audioDuration: Double, phaseTimings: PhaseTimings = PhaseTimings()) {
            self.fillers = fillers
            self.audioDuration = audioDuration
            self.phaseTimings = phaseTimings
        }
    }

    // MARK: - State

    private let computeUnits: ComputeUnits
    private let localModelURL: URL?
    private let labeler: FillerTypeClassifier?
    private let detectorLock = NSLock()
    private var detectorTask: Task<FillerDetector, Error>?
    private var detectorGeneration = 0

    /// Download-progress callback: receives a fraction in `0...1` while the
    /// model is fetched.
    public typealias Progress = ModelStore.Progress

    // MARK: - Model management

    /// Model input rate: 16 kHz mono, one frame every 20 ms.
    private static let sampleRate = 16_000

    // HF repo name and the compiled-model directory hosted inside it. Shared by
    // `init` and the prewarm API so there's one source of truth.
    private static let repoName = "uhm"
    private static let modelDirName = "uhm.mlmodelc"
    private static func makeStore() -> ModelStore { ModelStore(.init(name: repoName)) }

    /// Fetch the model snapshot (idempotent) and return the local `.mlmodelc` URL.
    private static func fetchModel(progress: Progress?) async throws -> URL {
        // Hosted on HF as an unzipped `uhm.mlmodelc/` directory; the Hub
        // snapshot API fetches it with per-file diffing and returns the repo
        // root. We load the compiled `.mlmodelc` directly (no on-device recompile).
        let snapshotURL = try await makeStore().snapshot(matching: ["\(modelDirName)/*"], progress: progress)
        return snapshotURL.appendingPathComponent(modelDirName, isDirectory: true)
    }

    /// Download and cache the model without constructing a detector.
    ///
    /// Call this at app launch (or behind a Wi-Fi/“prepare” gate) to prewarm the
    /// on-device model so the first real `analyze(...)` is instant. Safe to call
    /// repeatedly: once cached it's a cheap no-op, and the Hub snapshot only
    /// re-fetches files that actually changed. Concurrent calls are fine.
    ///
    /// - Parameter progress: Optional download progress in `0...1`.
    /// - Returns: The local file URL of the cached `uhm.mlmodelc`.
    /// - Throws: A download error if the model isn't cached and can't be fetched.
    @discardableResult
    public static func downloadModel(progress: Progress? = nil) async throws -> URL {
        try await fetchModel(progress: progress)
    }

    /// Whether the model is already downloaded and cached locally.
    ///
    /// Checked on disk with no network access, so it's safe to read on launch to
    /// decide whether to show a download UI before calling `downloadModel(progress:)`.
    public static var isModelDownloaded: Bool {
        guard let root = makeStore().cachedSnapshot() else { return false }
        return FileManager.default.fileExists(
            atPath: root.appendingPathComponent(modelDirName, isDirectory: true).path)
    }

    // MARK: - Init

    /// Creates an analyzer.
    ///
    /// Construction does **no** network or model I/O. The model is downloaded
    /// (if not already cached) and loaded lazily on the first `analyze(...)`.
    /// To fetch it ahead of time — e.g. at app launch — call
    /// `downloadModel(progress:)`.
    ///
    /// - Parameter computeUnits: Core ML compute-unit policy. Default `.all`.
    public init(computeUnits: ComputeUnits = .all) {
        self.computeUnits = computeUnits
        self.localModelURL = nil
        self.labeler = try? FillerTypeClassifier()  // bundled (25 KB), offline
    }

    /// Bench / power-user initializer — use a local model file (`.mlpackage` or
    /// `.mlmodelc`) instead of the downloaded default. Lets a host app ship its
    /// own variant. The model is loaded lazily on the first `analyze(...)`.
    ///
    /// - Parameters:
    ///   - modelURL: Local `.mlpackage` or `.mlmodelc` to load.
    ///   - computeUnits: Core ML compute-unit policy. Default `.all`.
    public init(modelURL: URL, computeUnits: ComputeUnits = .all) {
        self.computeUnits = computeUnits
        self.localModelURL = modelURL
        self.labeler = try? FillerTypeClassifier()
    }

    /// Lazily builds and memoizes the detector, downloading the model on first
    /// use when it isn't a local file. Concurrent first calls share one build;
    /// a failed build isn't cached, so the next `analyze(...)` retries.
    private func loadedDetector() async throws -> FillerDetector {
        detectorLock.lock()
        if let task = detectorTask {
            detectorLock.unlock()
            return try await task.value
        }
        let cu = computeUnits
        let local = localModelURL
        detectorGeneration += 1
        let generation = detectorGeneration
        let task = Task<FillerDetector, Error> {
            let modelURL: URL
            if let local {
                modelURL = local
            } else {
                modelURL = try await Self.fetchModel(progress: nil)
            }
            let cfg = MLModelConfiguration()
            cfg.computeUnits = cu.mlValue
            return try FillerDetector(modelPath: modelURL.path, configuration: cfg)
        }
        detectorTask = task
        detectorLock.unlock()
        do {
            return try await task.value
        } catch {
            // Don't cache a failed build — let the next analyze retry — but only
            // clear if no newer build has replaced ours.
            detectorLock.lock()
            if detectorGeneration == generation { detectorTask = nil }
            detectorLock.unlock()
            throw error
        }
    }

    // MARK: - Analyze

    /// Detects filler words in an audio file at the given path.
    ///
    /// Any format `AVAudioFile` can read is accepted; audio is decoded to
    /// 16 kHz mono internally.
    ///
    /// - Parameters:
    ///   - audioPath: Filesystem path to the audio file.
    ///   - options: Bias, type labeling, and duration thresholds. Default `.default`.
    ///   - progressHandler: Optional inference progress in `0...1`.
    /// - Returns: The detected fillers plus timing metadata.
    /// - Throws: A decode/model error, or `CancellationError` if the enclosing task is cancelled.
    public func analyze(
        audioPath: String,
        options: Options = .default,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> Result {
        var timings = PhaseTimings()
        // Bail early if the caller already cancelled.
        try Task.checkCancellation()
        // Download (first call only) + load the model. No-op once cached/loaded.
        let detector = try await loadedDetector()

        // Decode once, at 16 kHz mono, and share the samples between the frame
        // detector and the type labeler.
        let tDecode = CFAbsoluteTimeGetCurrent()
        let samples = try AudioIO.loadMono(path: audioPath, sampleRate: Double(Self.sampleRate))
        timings.decodeSec = CFAbsoluteTimeGetCurrent() - tDecode
        let audioDur = Double(samples.count) / Double(Self.sampleRate)
        let minConf = options.minConfidence ?? options.bias.minConfidence

        var detTimings = FillerDetector.Timings()
        var detections = try detector.detect(
            samples: samples,
            progressHandler: progressHandler,
            timingsHandler: { detTimings = $0 }
        )
        .filter { $0.confidence >= minConf && $0.duration >= options.minDurationSec }
        .map { Detection(start: $0.start, end: $0.end, confidence: $0.confidence, type: nil) }
        timings.inferenceSec = detTimings.inferenceSec
        timings.prepSec = detTimings.prepSec
        timings.groupSec = detTimings.groupSec

        if options.includeTypes, labeler != nil {
            try Task.checkCancellation()
            let tLabel = CFAbsoluteTimeGetCurrent()
            detections = try await labelDetections(detections, samples: samples)
            timings.labelingSec = CFAbsoluteTimeGetCurrent() - tLabel
        }

        return Result(fillers: detections, audioDuration: audioDur, phaseTimings: timings)
    }

    /// Detects filler words in the audio at the given file URL.
    ///
    /// - Parameters:
    ///   - audioURL: File URL of the audio. Any format `AVAudioFile` can read.
    ///   - options: Bias, type labeling, and duration thresholds. Default `.default`.
    ///   - progressHandler: Optional inference progress in `0...1`.
    /// - Returns: The detected fillers plus timing metadata.
    public func analyze(
        audioURL: URL,
        options: Options = .default,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> Result {
        try await analyze(audioPath: audioURL.path, options: options, progressHandler: progressHandler)
    }

    /// Detects filler words in raw PCM samples.
    ///
    /// - Parameters:
    ///   - samples: Mono PCM samples. Resampled to 16 kHz internally if needed.
    ///   - sampleRate: Sample rate of `samples`, in Hz.
    ///   - options: Bias, type labeling, and duration thresholds. Default `.default`.
    ///   - progressHandler: Optional inference progress in `0...1`.
    /// - Returns: The detected fillers plus timing metadata.
    public func analyze(
        samples: [Float],
        sampleRate: Int,
        options: Options = .default,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> Result {
        let tmpURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("uhm-input-\(UUID().uuidString).wav")
        try AudioIO.writeWav(samples: samples, sampleRate: sampleRate, to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }
        return try await analyze(audioURL: tmpURL, options: options, progressHandler: progressHandler)
    }

    // MARK: - Private

    private func labelDetections(_ detections: [Detection], samples: [Float]) async throws -> [Detection] {
        guard let labeler = self.labeler else { return detections }
        let window = Self.sampleRate  // 1 s clip centered on each detection
        var out: [Detection] = []
        for d in detections {
            // Bail between detections so cancellation lands promptly when the
            // caller flips away (e.g. a setting change invalidates these fillers).
            try Task.checkCancellation()
            let center = Int((d.start + d.end) / 2 * Double(Self.sampleRate))
            let start = max(0, center - window / 2)
            let end = min(samples.count, start + window)
            var clip = Array(samples[start..<end])
            if clip.count < window {
                clip.append(contentsOf: [Float](repeating: 0, count: window - clip.count))
            }
            let tmpURL = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("uhm-label-\(UUID().uuidString).wav")
            try AudioIO.writeWav(samples: clip, sampleRate: Self.sampleRate, to: tmpURL)
            defer { try? FileManager.default.removeItem(at: tmpURL) }
            let labels = try labeler.detect(audioPath: tmpURL.path)
            let best = labels.max { $0.confidence < $1.confidence }
            let type = best.flatMap { FillerType(rawValue: $0.label) }
            out.append(Detection(start: d.start, end: d.end, confidence: d.confidence, type: type))
        }
        return out
    }
}
