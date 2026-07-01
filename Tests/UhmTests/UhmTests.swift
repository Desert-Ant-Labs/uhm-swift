import XCTest
@testable import Uhm

final class UhmTests: XCTestCase {

    func testBiasThresholds() {
        XCTAssertEqual(Uhm.Bias.precision.minConfidence, 0.75)
        XCTAssertEqual(Uhm.Bias.balanced.minConfidence, 0.65)
        XCTAssertEqual(Uhm.Bias.recall.minConfidence, 0.50)
    }

    func testDefaultOptions() {
        let options = Uhm.Options.default
        XCTAssertEqual(options.bias.minConfidence, 0.65)
        XCTAssertTrue(options.includeTypes)
        XCTAssertNil(options.minConfidence)
        XCTAssertEqual(options.minDurationSec, 0.12)
    }
}

final class WordReconciliationTests: XCTestCase {

    private func filler(_ start: Double, _ end: Double) -> Uhm.Detection {
        Uhm.Detection(start: start, end: end, confidence: 1, type: .um)
    }

    func testNoFillersPassesWordsThrough() {
        let words = [WordRange(start: 1, end: 2, word: "b"),
                     WordRange(start: 0, end: 1, word: "a")]
        let out = Uhm.reconcileWords(words, fillers: [])
        XCTAssertEqual(out.map(\.word), ["a", "b"])  // sorted by start
    }

    func testWordFullyInsideFillerIsDropped() {
        let words = [WordRange(start: 1.0, end: 1.3, word: "um")]
        let out = Uhm.reconcileWords(words, fillers: [filler(0.5, 2.0)])
        XCTAssertTrue(out.isEmpty)
    }

    func testWordLeadingIntoFillerIsTrimmed() {
        let words = [WordRange(start: 0.0, end: 1.2, word: "so")]
        let out = Uhm.reconcileWords(words, fillers: [filler(1.0, 2.0)])
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].start, 0.0, accuracy: 1e-9)
        XCTAssertEqual(out[0].end, 1.0, accuracy: 1e-9)
    }

    func testWordTrailingOutOfFillerIsPushed() {
        let words = [WordRange(start: 1.5, end: 2.5, word: "then")]
        let out = Uhm.reconcileWords(words, fillers: [filler(1.0, 2.0)])
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].start, 2.0, accuracy: 1e-9)
        XCTAssertEqual(out[0].end, 2.5, accuracy: 1e-9)
    }

    func testNonOverlappingWordUnchanged() {
        let words = [WordRange(start: 3.0, end: 4.0, word: "clear")]
        let out = Uhm.reconcileWords(words, fillers: [filler(1.0, 2.0)])
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].start, 3.0, accuracy: 1e-9)
        XCTAssertEqual(out[0].end, 4.0, accuracy: 1e-9)
    }
}
