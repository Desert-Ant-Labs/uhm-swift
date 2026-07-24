# Uhm: On-device Filler-Word Detection for Swift (iOS, macOS)

On-device filler-word detection for iOS and macOS. A frame-precise classifier finds "uh", "um", "hmm" and other fillers in audio with one prediction every 20 ms. Trained on English; transfers acoustically to Spanish, French, German, and Dutch without retraining.

Uhm does acoustic disfluency detection: it finds the disfluencies directly from the waveform, so there is no transcript to run first. That makes it a good fit for editing workflows that need to remove filler words, from podcast editing to general transcription cleanup, all without sending audio off the device.

```swift
import Uhm

let uhm = Uhm()                                  // cheap; no I/O
let result = try await uhm.analyze(audioURL: url) // downloads + loads on first call
for f in result.fillers {
    print(f.start, f.end, f.type ?? .other)
}
```

Constructing `Uhm()` does no network or model work. The compiled Core ML model (`uhm.mlmodelc`) downloads on the first `analyze(...)` (or when you call `downloadModel`) from [`huggingface.co/desert-ant-labs/uhm`](https://huggingface.co/desert-ant-labs/uhm) and caches in Application Support. Downloads go through the Hugging Face Hub snapshot API (via [`swift-transformers`](https://github.com/huggingface/swift-transformers), wrapped by [`desert-ant-swift`](https://github.com/Desert-Ant-Labs/desert-ant-swift)'s `ModelStore`), so model updates only re-fetch the files that actually changed.

## Prewarming the model

The model is fetched lazily on the first `analyze(...)`. To get that out of the way ahead of time (e.g. at app launch or behind a “prepare” screen, and to show a progress bar), call `downloadModel`, optionally gated on `isModelDownloaded`:

```swift
if !Uhm.isModelDownloaded {          // on-disk check, no network
    try await Uhm.downloadModel { fraction in
        print("model download \(Int(fraction * 100))%")
    }
}

// First analyze is now instant; the model is already cached.
let uhm = Uhm()
let result = try await uhm.analyze(audioURL: url)
```

`downloadModel` is idempotent and safe to call repeatedly; once cached it's a cheap no-op and only re-fetches files that changed. If you skip it, the first `analyze(...)` performs the download itself (without a progress callback).

## Install

```swift
.package(url: "https://github.com/Desert-Ant-Labs/uhm-swift.git", from: "0.1.0")
// product: "Uhm"
```

## API

```swift
// Defaults: balanced bias (0.65 confidence threshold), filler typing on.
let uhm = Uhm()

// Tune precision/recall for the use case.
let result = try await uhm.analyze(
    audioURL: url,
    options: .init(bias: .precision)
)

// Or work on float samples directly.
let result = try await uhm.analyze(samples: floats, sampleRate: 16_000)
```

Bias presets:

| Bias | Threshold | Use |
|---|---:|---|
| `.precision` | 0.75 | Avoid false positives; safest for automatic cuts. |
| `.balanced` | 0.65 | Default. Clean cuts on the labeled corpus. |
| `.recall` | 0.50 | Catch more; review downstream. |

Set `includeTypes: false` in `Uhm.Options` to skip the bundled type labeler when you only need filler vs. not-filler spans.

## Example App

A minimal SwiftUI example is included in `Examples/UhmExample`. Pick an audio file, run detection, and list the fillers with timestamps. The model downloads on first run and caches locally.

## Other platforms

- Model weights and card: [`desert-ant-labs/uhm`](https://huggingface.co/desert-ant-labs/uhm)
- Kotlin and JS SDKs are planned.

## License

[Desert Ant Labs Source-Available License](https://license.desertant.com/1.0). Free for
most apps; a commercial license is required at scale. Full terms are at the link.
Licensing: <licensing@desertant.com>.
