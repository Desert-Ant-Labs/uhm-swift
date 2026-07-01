import SwiftUI
import UniformTypeIdentifiers
import Uhm

// Minimal Uhm example: pick an audio file, run detection, list the fillers
// with click-to-copy timestamps. The model downloads once on first run and
// caches locally.
struct ContentView: View {

    var body: some View {
        NavigationStack {
            Group {
                switch state {
                case .idle:        idleState
                case .downloading: busy("Downloading model…")
                case .analyzing:   busy("Analyzing…")
                case .results:     results
                case .failed(let message): failed(message)
                }
            }
            .navigationTitle("Uhm")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Pick audio", systemImage: "waveform") {
                        picking = true
                    }
                    .disabled(state == .downloading || state == .analyzing)
                }
            }
            .fileImporter(isPresented: $picking,
                          allowedContentTypes: [.audio],
                          allowsMultipleSelection: false) { result in
                if case let .success(urls) = result, let url = urls.first {
                    analyze(url)
                }
            }
        }
    }

    private enum State: Equatable {
        case idle, downloading, analyzing, results, failed(String)
    }

    @State private var state: State = .idle
    @State private var picking = false
    @State private var fillers: [Uhm.Detection] = []
    @State private var audioDuration: Double = 0

    private func analyze(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        Task {
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            do {
                // Prewarm with a progress bar the first time; instant afterwards.
                if !Uhm.isModelDownloaded {
                    state = .downloading
                    try await Uhm.downloadModel { _ in }
                }
                state = .analyzing
                let uhm = Uhm()
                let result = try await uhm.analyze(audioURL: url)
                fillers = result.fillers
                audioDuration = result.audioDuration
                state = .results
            } catch is CancellationError {
                state = .idle
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }

    private var idleState: some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform")
                .font(.system(size: 56))
                .foregroundStyle(.tint)
                .opacity(0.5)
            Text("Pick an audio file to find filler words")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }

    private func busy(_ label: String) -> some View {
        VStack(spacing: 16) {
            ProgressView()
            Text(label).foregroundStyle(.secondary)
        }
    }

    private func failed(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 44))
                .foregroundStyle(.orange)
            Text(message)
                .font(.system(size: 14))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }

    private var results: some View {
        List {
            Section {
                ForEach(Array(fillers.enumerated()), id: \.offset) { _, f in
                    HStack {
                        Text(f.type.map { "\($0)" } ?? "filler")
                            .font(.system(.body, design: .rounded).weight(.medium))
                        Spacer()
                        Text(timecode(f.start))
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
            } header: {
                Text("\(fillers.count) filler\(fillers.count == 1 ? "" : "s") · \(timecode(audioDuration)) of audio")
            }
        }
    }

    private func timecode(_ seconds: Double) -> String {
        let m = Int(seconds) / 60
        let s = seconds - Double(m * 60)
        return String(format: "%d:%05.2f", m, s)
    }
}

#Preview {
    ContentView()
}
