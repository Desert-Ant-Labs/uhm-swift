import CoreML
import Foundation
import SoundAnalysis

/// Per-filler type labeler. A small sound classifier (`uh` / `um` / `hmm` /
/// `other`) trained with CreateML's MLSoundClassifier, so inference runs
/// through Apple's SoundAnalysis framework using the built-in audio feature
/// extractor — the bundled model is just the classifier head. Used by `Uhm`
/// to attach a `FillerType` to each detection from the frame-level
/// `FillerDetector`.
final class FillerTypeClassifier: NSObject, @unchecked Sendable {
    struct Config: Sendable {
        var minConfidence: Double
        var mergeGap: Double           // merge adjacent filler windows within this gap (s)
        var minDuration: Double        // discard merged events shorter than this
        var overlapFactor: Double      // SNClassifySoundRequest window overlap
        var fillerLabel: String

        static let `default` = Config(
            minConfidence: 0.6,
            mergeGap: 0.20,
            minDuration: 0.18,
            overlapFactor: 0.5,
            fillerLabel: "filler"
        )
    }

    let config: Config
    private let model: MLModel

    init(modelURL: URL? = nil, config: Config = .default) throws {
        self.config = config
        let url = try modelURL ?? Self.resolveBundledModel()
        let compiled: URL
        if url.pathExtension == "mlmodelc" {
            compiled = url
        } else {
            compiled = try MLModel.compileModel(at: url)
        }
        self.model = try MLModel(contentsOf: compiled)
    }

    func detect(audioPath: String) throws -> [Filler] {
        let url = URL(fileURLWithPath: audioPath)
        let analyzer = try SNAudioFileAnalyzer(url: url)
        let request = try SNClassifySoundRequest(mlModel: model)
        request.overlapFactor = config.overlapFactor

        let observer = ResultsObserver(fillerLabel: config.fillerLabel,
                                       minConfidence: config.minConfidence)
        try analyzer.add(request, withObserver: observer)
        analyzer.analyze()

        return mergeAdjacent(observer.hits)
    }

    // MARK: - Bundled model resolution

    private static func resolveBundledModel() throws -> URL {
        let candidates = ["UhmLabel.mlmodelc", "UhmLabel.mlmodel"]
        for name in candidates {
            let nameOnly = (name as NSString).deletingPathExtension
            let ext = (name as NSString).pathExtension
            // The Resources folder in SwiftPM is namespaced under "Resources"
            for subdir in [nil, "Resources"] {
                if let url = Bundle.module.url(forResource: nameOnly, withExtension: ext, subdirectory: subdir) {
                    return url
                }
            }
        }
        throw NSError(domain: "FillerTypeClassifier", code: 404, userInfo: [
            NSLocalizedDescriptionKey: "Bundled UhmLabel model not found in package Resources."
        ])
    }

    // MARK: - Post-processing

    private struct Hit {
        let start: Double
        let end: Double
        let confidence: Double
        let label: String
    }

    private func mergeAdjacent(_ hits: [Hit]) -> [Filler] {
        guard !hits.isEmpty else { return [] }
        let sorted = hits.sorted { $0.start < $1.start }
        var merged: [(start: Double, end: Double, conf: Double, n: Int, label: String)] = []
        for h in sorted {
            // Merge only when adjacent AND same label, so per-type stays distinct
            if var last = merged.last, h.start - last.end <= config.mergeGap, last.label == h.label {
                last.end = max(last.end, h.end)
                last.conf += h.confidence
                last.n += 1
                merged[merged.count - 1] = last
            } else {
                merged.append((h.start, h.end, h.confidence, 1, h.label))
            }
        }
        return merged.compactMap { run in
            let dur = run.end - run.start
            guard dur >= config.minDuration else { return nil }
            // Strip "filler_" prefix for output: "filler_uh" → "uh"
            let displayLabel = run.label.hasPrefix("filler_")
                ? String(run.label.dropFirst("filler_".count))
                : run.label
            return Filler(
                label: displayLabel,
                start: run.start,
                end: run.end,
                confidence: run.conf / Double(run.n)
            )
        }
    }

    // MARK: - SoundAnalysis observer

    private final class ResultsObserver: NSObject, SNResultsObserving {
        let fillerLabel: String
        let minConfidence: Double
        var hits: [Hit] = []

        init(fillerLabel: String, minConfidence: Double) {
            self.fillerLabel = fillerLabel
            self.minConfidence = minConfidence
        }

        func request(_ request: SNRequest, didProduce result: SNResult) {
            guard let r = result as? SNClassificationResult else { return }
            guard let top = r.classifications.first else { return }
            // Accept:
            //   - "filler" (binary detector)
            //   - "filler_*" (per-type detector with prefix)
            //   - "uh"/"um"/"hmm"/"other" (Uhm-Label: forced filler-only model)
            let label = top.identifier
            let knownFillerWords: Set<String> = ["uh", "um", "hmm", "other", "ah", "eh", "er", "mm"]
            let isFiller = label == fillerLabel
                        || label.hasPrefix("filler_")
                        || knownFillerWords.contains(label)
            guard isFiller, Double(top.confidence) >= minConfidence else { return }
            hits.append(Hit(
                start: r.timeRange.start.seconds,
                end: (r.timeRange.start + r.timeRange.duration).seconds,
                confidence: Double(top.confidence),
                label: label
            ))
        }

        func request(_ request: SNRequest, didFailWithError error: Error) {}
        func requestDidComplete(_ request: SNRequest) {}
    }
}
