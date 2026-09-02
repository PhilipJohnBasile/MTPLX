import XCTest
@testable import MTPLXAppCore

// MARK: - StreamingPerfRegressionTests
//
// Pins the 2026-08-17 streaming-freeze fixes: per-frame render cost
// must stay O(blocks), the typewriter can never paste an unbounded
// backlog in one frame, and the metrics SSE framing must keep exact
// message boundaries. Each test guards a specific mechanism that let
// "freeze → vomit" ship while every engine-side number looked healthy.

final class StreamingPerfRegressionTests: XCTestCase {

    // MARK: fenceCount (utf8 rewrite) parity

    /// Reference implementation: the original Character walk.
    private func referenceFenceCount(_ text: String) -> Int {
        var count = 0
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("```") {
                count += 1
                index = text.index(index, offsetBy: 3)
                continue
            }
            index = text.index(after: index)
        }
        return count
    }

    func testFenceCountMatchesReferenceWalk() {
        let samples = [
            "",
            "```",
            "``",
            "````",       // 4 backticks: one fence, one leftover
            "``````",     // 6 backticks: two fences
            "`````",      // 5 backticks: one fence
            "no fences at all",
            "prefix ```swift\ncode\n``` suffix",
            "emoji 🐦🎮 before ``` and after",
            "backtick ` single ` pairs `` still ``",
            "日本語テキスト```コード```終わり",
            String(repeating: "`", count: 31),
            "```one``` middle ```two``` ```three```",
        ]
        for sample in samples {
            XCTAssertEqual(
                StreamingMarkdownBlockSafety.fenceCount(in: sample),
                referenceFenceCount(sample),
                "fenceCount diverged for: \(sample.debugDescription)"
            )
        }
    }

    func testBlockCarriesCachedRenderMetrics() {
        let block = StreamingDocumentBlock(
            id: 7,
            text: "a ```\nb ``` c",
            kind: .plain,
            finalized: true
        )
        XCTAssertEqual(block.fenceMarkerCount, 2)
        XCTAssertEqual(block.lineCount, 2)
    }

    func testBlockClassificationMatchesTextClassification() {
        let texts = [
            "prose line",
            "```swift",
            "let x = 1",
            "print(x)",
            "```",
            "after the fence",
            "inline ``` odd fence prose",
            "tail line",
        ]
        let blocks = texts.enumerated().map { index, text in
            StreamingDocumentBlock(
                id: index,
                text: text,
                kind: .plain,
                finalized: index < texts.count - 1
            )
        }
        let fromTexts = StreamingMarkdownBlockSafety.classifyRoles(texts)
        let fromBlocks = StreamingMarkdownBlockSafety.classifyRoles(blocks)
        XCTAssertEqual(fromTexts, fromBlocks)
    }

    // MARK: line-segment coalescing still fires with the counter gate

    @MainActor
    func testLineCoalescingStillMergesWithCandidateGate() {
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 8
        defer { StreamingDocumentStore.lineSegmentSizeOverrideForTesting = nil }
        let store = StreamingDocumentStore(mode: .plainLines)
        for lineNumber in 0..<40 {
            store.append("line number \(lineNumber)\n")
        }
        XCTAssertGreaterThan(store.liveSegmentMergeCount, 0,
            "candidate gate must not starve the coalescer")
        XCTAssertLessThan(store.blocks.count, 40,
            "merges must keep realized block count sublinear in lines")
        // The document text survives merging byte for byte.
        XCTAssertEqual(
            store.blocks.map(\.text).joined(separator: "\n"),
            (0..<40).map { "line number \($0)" }.joined(separator: "\n")
        )
        XCTAssertEqual(
            store.rawText,
            (0..<40).map { "line number \($0)\n" }.joined()
        )
    }

    @MainActor
    func testFenceLinesAreNeverMerged() {
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 4
        defer { StreamingDocumentStore.lineSegmentSizeOverrideForTesting = nil }
        let store = StreamingDocumentStore(mode: .plainLines)
        store.append("```swift\n")
        for lineNumber in 0..<30 {
            store.append("code \(lineNumber)\n")
        }
        store.append("```\n")
        for block in store.blocks where block.text.contains("\n") {
            XCTAssertFalse(block.text.contains("```"),
                "merged segment may never contain a fence line")
        }
    }

    // MARK: typewriter reveal ceiling (the anti-vomit bound)

    @MainActor
    func testPacedCutBoundsSingleFrameReveal() {
        // Even a runaway budget must respect the frame ceiling — the
        // unbounded whole-drain WAS the "vomit" paste.
        let backlog = String(repeating: "x", count: 10_000)
        let (reveal, rest) = ChatViewModel.pacedCut(backlog, budget: 10_000)
        XCTAssertLessThanOrEqual(reveal.count, 256,
            "a stalled-then-recovered stream must catch up as fast typing, not one paste")
        XCTAssertEqual(reveal + rest, backlog, "no bytes may be lost or reordered")
    }

    @MainActor
    func testPacedCutKeepsTypingAliveOnZeroBudget() {
        // While the arrival-rate EMA warms up the budget can be 0; the
        // floor keeps characters flowing instead of freezing the reveal.
        let backlog = String(repeating: "y", count: 100)
        let (reveal, rest) = ChatViewModel.pacedCut(backlog, budget: 0)
        XCTAssertEqual(reveal.count, 3)
        XCTAssertEqual(reveal + rest, backlog)
    }

    @MainActor
    func testPacedCutDrainsSmallBuffersWhole() {
        let small = "ab"
        let (reveal, rest) = ChatViewModel.pacedCut(small, budget: 0)
        XCTAssertEqual(reveal, small)
        XCTAssertEqual(rest, "")
    }

    // MARK: typewriter pacer (a copy-lane burst types across the gap)

    private struct PacerTick {
        let tick: Int
        /// Unrevealed characters when the tick began.
        let pending: Int
        let revealed: Int
        let backlogAfter: Int
        /// The pacer had seen two arrivals (an inter-arrival estimate).
        let estimatorWarm: Bool
    }

    private struct PacerRun {
        var ticks: [PacerTick] = []
        /// Unrevealed characters just before each arrival after the first.
        var backlogBeforeArrival: [Int] = []
        var fed = ""
        var revealed = ""
    }

    /// Replays the display tick exactly as `flushStreamingBuffers` does
    /// (pacer budget, then `pacedCut`) on a simulated uptime clock:
    /// `chunk` characters land every `gap` seconds and the reveal ticks
    /// at 60 Hz. The stream end drains whatever is left, as finalize,
    /// cancel and the stream return do.
    @MainActor
    private func replayPacer(chunk: Int, gap: Double, chunks: Int) -> PacerRun {
        var pacer = StreamTypewriterPacer()
        var run = PacerRun()
        var buffer = ""
        let tickSeconds = 1.0 / 60.0
        let totalTicks = Int((Double(chunks) * gap / tickSeconds).rounded(.up))
        var delivered = 0
        for tick in 0..<totalTicks {
            let now = Double(tick) * tickSeconds
            // Deliver every chunk due by this tick; the SSE task lands
            // ahead of the frame that reveals it.
            while delivered < chunks, Double(delivered) * gap <= now + 1e-9 {
                if delivered > 0 { run.backlogBeforeArrival.append(buffer.count) }
                let letter = Character(UnicodeScalar(UInt8(97 + delivered % 26)))
                let text = String(repeating: letter, count: chunk)
                buffer += text
                run.fed += text
                pacer.recordArrival(chars: chunk, now: Double(delivered) * gap)
                delivered += 1
            }
            guard delivered > 0 else { continue }
            let pending = buffer.count
            var revealedNow = 0
            if buffer.isEmpty {
                pacer.noteIdleTick(now: now)
            } else {
                let budget = pacer.tickBudget(backlog: pending, now: now)
                let cut = ChatViewModel.pacedCut(buffer, budget: budget)
                run.revealed += cut.reveal
                buffer = cut.rest
                revealedNow = cut.reveal.count
            }
            run.ticks.append(PacerTick(
                tick: tick,
                pending: pending,
                revealed: revealedNow,
                backlogAfter: buffer.count,
                estimatorWarm: pacer.expectedGapSeconds > 0
            ))
        }
        run.revealed += buffer
        return run
    }

    @MainActor
    func testPacerSpreadsCopyLaneBurstAcrossTheRoundGap() {
        // Context-copy lane: one ~110-character block per engine round,
        // every 250 ms. The old estimator pasted each block in one frame.
        let chunk = 110
        let run = replayPacer(chunk: chunk, gap: 0.25, chunks: 8)
        let warm = run.ticks.filter(\.estimatorWarm)
        XCTAssertFalse(warm.isEmpty)
        // (a) No frame pastes more than about a quarter of a block once
        //     the estimator has two samples.
        for sample in warm {
            XCTAssertLessThanOrEqual(sample.revealed, chunk / 4,
                "tick \(sample.tick) revealed \(sample.revealed) characters: a paste, not typing")
        }
        // The block is typed across the whole gap: nearly every frame
        // between two rounds reveals something.
        let steady = run.ticks.filter { $0.estimatorWarm && $0.tick >= 30 }
        let flowing = steady.filter { $0.revealed > 0 }.count
        XCTAssertGreaterThanOrEqual(flowing, steady.count * 9 / 10,
            "the reveal must flow through the gap rather than idle after a paste")
        // (b) The backlog is gone by the time the next block lands
        //     (within one frame's reveal).
        let maxTickReveal = warm.map(\.revealed).max() ?? 0
        XCTAssertGreaterThanOrEqual(run.backlogBeforeArrival.count, 7)
        for backlog in run.backlogBeforeArrival.dropFirst() {
            XCTAssertLessThanOrEqual(backlog, maxTickReveal,
                "a block must finish typing as the next one arrives, not pile up")
        }
        // (d) The end-of-stream drain reveals everything, in order.
        XCTAssertEqual(run.revealed, run.fed)
    }

    @MainActor
    func testPacerKeepsTokenStreamRevealingEveryTickWithoutLag() {
        // (c) Plain token-by-token arrival: ~5 characters every 42 ms.
        let chunk = 5
        let run = replayPacer(chunk: chunk, gap: 0.042, chunks: 60)
        XCTAssertFalse(run.ticks.isEmpty)
        XCTAssertGreaterThan(run.ticks.filter { $0.pending > 0 }.count, 60)
        for sample in run.ticks {
            if sample.pending > 0 {
                XCTAssertGreaterThan(sample.revealed, 0,
                    "tick \(sample.tick) had text pending and revealed nothing")
            }
            XCTAssertLessThanOrEqual(sample.backlogAfter, chunk,
                "tick \(sample.tick) let the backlog grow past one chunk")
        }
        for backlog in run.backlogBeforeArrival {
            XCTAssertLessThanOrEqual(backlog, chunk,
                "steady arrival must not accumulate lag")
        }
        XCTAssertEqual(run.revealed, run.fed)
    }

    @MainActor
    func testPacerBudgetIsZeroWithoutBacklogAndColdEstimatorFloorsAtThree() {
        var pacer = StreamTypewriterPacer()
        XCTAssertEqual(pacer.tickBudget(backlog: 0, now: 1.0), 0)
        pacer.recordArrival(chars: 110, now: 1.0)
        // One sample: no rate, no gap estimate; the caller's floor of
        // three characters keeps the first block typing.
        let budget = pacer.tickBudget(backlog: 110, now: 1.0 + 1.0 / 60.0)
        XCTAssertEqual(budget, 0)
        let (reveal, rest) = ChatViewModel.pacedCut(String(repeating: "z", count: 110), budget: budget)
        XCTAssertEqual(reveal.count, 3)
        XCTAssertEqual(rest.count, 107)
    }

    // MARK: SSE line accumulator framing

    private func messages(from payload: String) -> [SSEMessage] {
        var accumulator = SSELineAccumulator()
        var out: [SSEMessage] = []
        for byte in payload.utf8 {
            if let message = accumulator.consume(byte) {
                out.append(message)
            }
        }
        return out
    }

    func testAccumulatorFramesCRLFAndLFMessages() {
        let payload = "event: snapshot\r\ndata: {\"a\":1}\r\n\r\n"
            + ": heartbeat comment\n"
            + "event: progress\ndata: {\"b\":2}\n\n"
            + "data: first\ndata: second\n\n"
        let parsed = messages(from: payload)
        XCTAssertEqual(parsed, [
            SSEMessage(event: "snapshot", data: "{\"a\":1}"),
            SSEMessage(event: "progress", data: "{\"b\":2}"),
            SSEMessage(event: "message", data: "first\nsecond"),
        ])
    }

    func testAccumulatorMatchesLegacyParser() {
        let payload = "event: thermal\ndata: {\"t\":61.5}\n\n"
            + "event: new_max_tps\r\ndata: {\"tps\":81.2}\r\n\r\n"
        let legacy = SSEParser().parse(payload)
        XCTAssertEqual(messages(from: payload), legacy)
    }

    func testAccumulatorHoldsIncompleteMessage() {
        // No trailing blank line: nothing may be emitted early.
        XCTAssertTrue(messages(from: "event: x\ndata: 1\n").isEmpty)
    }
}
