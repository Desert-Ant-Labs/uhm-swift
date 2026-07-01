import AVFoundation
import Foundation

/// Shared audio helpers used across the pipeline: decode any
/// AVFoundation-readable file to mono `Float` samples, and write mono samples
/// back out as a 16-bit PCM WAV (for the SoundAnalysis-based type labeler,
/// which only reads from a file URL).
enum AudioIO {

    /// Decode the file at `path` to mono `Float` samples at `sampleRate`.
    /// Returns an empty array if the file can't be opened or converted.
    static func loadMono(path: String, sampleRate: Double) throws -> [Float] {
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
        let inFormat = file.processingFormat
        guard
            let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                          sampleRate: sampleRate,
                                          channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: inFormat, to: outFormat),
            let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat,
                                            frameCapacity: AVAudioFrameCount(file.length))
        else { return [] }

        try file.read(into: inBuffer)

        let capacity = AVAudioFrameCount(Double(inBuffer.frameLength) * (sampleRate / inFormat.sampleRate) + 1024)
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return [] }

        var fed = false
        var error: NSError?
        converter.convert(to: outBuffer, error: &error) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return inBuffer
        }
        if let error { throw error }

        guard let channel = outBuffer.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: channel, count: Int(outBuffer.frameLength)))
    }

    /// Write mono `samples` to a 16-bit PCM WAV at `url`.
    static func writeWav(samples: [Float], sampleRate: Int, to url: URL) throws {
        var data = Data()
        let pcm = samples
            .map { Int16(max(-1, min(1, $0)) * 32_767) }
            .withUnsafeBufferPointer { Data(buffer: $0) }
        let dataSize = UInt32(pcm.count)

        func append<T>(_ value: T) {
            var v = value
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }

        data.append(contentsOf: "RIFF".utf8); append(UInt32(36) + dataSize)
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8); append(UInt32(16))
        append(UInt16(1)); append(UInt16(1)); append(UInt32(sampleRate))
        append(UInt32(sampleRate * 2)); append(UInt16(2)); append(UInt16(16))
        data.append(contentsOf: "data".utf8); append(dataSize)
        data.append(pcm)
        try data.write(to: url)
    }
}
