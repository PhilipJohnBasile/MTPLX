import Foundation
import XCTest

@testable import MTPLXAppCore

// MARK: - ServerErrorChunkTests
//
// The daemon fails a request mid-stream with one frame:
// `choices[0].finish_reason == "error"` plus a top-level OpenAI-shape
// `error` object, then `[DONE]`. Before this fix the chunk decoder
// dropped the error object, the finish became an ordinary `.finished`,
// and the turn was persisted as if it had completed — no message, no
// Retry, and a bubble that read "Interrupted reply" at best. These pin
// the whole path: chunk -> event -> view-model failure -> persisted
// message the settled bubble can label.

final class ServerErrorChunkTests: XCTestCase {

    private let client = MTPLXChatClient(
        apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:1")!)
    )

    // MARK: Chunk fixtures

    /// The exact shape `mtplx/server/openai.py` emits from `error_chunk`.
    private static let errorFrame = """
    {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "m",
     "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
     "error": {"message": "Prompt of 140000 tokens exceeds the 131072-token context window.",
               "type": "invalid_request_error", "code": "HTTPException", "param": null}}
    """

    func testErrorFrameDecodesToServerErrorAndNeverToFinished() throws {
        let events = try XCTUnwrap(client.streamEvents(fromDataPayload: Self.errorFrame))
        XCTAssertEqual(events.count, 1)
        guard case .serverError(let message) = events[0] else {
            return XCTFail("expected .serverError, got \(events[0])")
        }
        XCTAssertEqual(
            message,
            "Prompt of 140000 tokens exceeds the 131072-token context window."
        )
    }

    func testErrorFrameWithNumericCodeStillDecodes() throws {
        // Other OpenAI-compatible servers put a number in `code`; a field
        // the app never reads must not be able to drop the whole frame.
        let frame = """
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
         "error": {"message": "out of memory", "type": "server_error", "code": 507}}
        """
        let events = try XCTUnwrap(client.streamEvents(fromDataPayload: frame))
        guard case .serverError(let message) = events.first else {
            return XCTFail("expected .serverError, got \(events)")
        }
        XCTAssertEqual(message, "out of memory")
    }

    func testErrorFinishWithoutErrorObjectStillFailsTheTurn() throws {
        let frame = """
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]}
        """
        let events = try XCTUnwrap(client.streamEvents(fromDataPayload: frame))
        guard case .serverError(let message) = events.first else {
            return XCTFail("a finish_reason of error must never become .finished; got \(events)")
        }
        XCTAssertFalse(message.isEmpty)
    }

    func testStopFrameIsUnchanged() throws {
        let frame = """
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11},
         "mtplx_stats": {"raw_decode_tok_s": 41.0}}
        """
        let events = try XCTUnwrap(client.streamEvents(fromDataPayload: frame))
        XCTAssertEqual(events.count, 1)
        guard case .finished(let reason, let usage, let stats) = events[0] else {
            return XCTFail("expected .finished, got \(events[0])")
        }
        XCTAssertEqual(reason, "stop")
        XCTAssertEqual(usage?.completionTokens, 8)
        XCTAssertEqual(stats?.rawDecodeTokS, 41.0)
    }

    func testContentFrameCarriesNoTerminalEvent() throws {
        let frame = """
        {"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]}
        """
        let events = try XCTUnwrap(client.streamEvents(fromDataPayload: frame))
        XCTAssertEqual(events.count, 1)
        guard case .contentDelta(let text) = events[0] else {
            return XCTFail("expected .contentDelta, got \(events[0])")
        }
        XCTAssertEqual(text, "hi")
    }

    // MARK: View model

    @MainActor
    func testServerFailureMidStreamRecordsErrorOffersRetryAndPersistsMessage() async throws {
        let daemon = try await ChatFakeDaemon.start(chatHandler: """
        self.wfile.write(sse({"id": "chatcmpl-fail",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
        self.wfile.write(sse({"id": "chatcmpl-fail",
            "choices": [{"index": 0, "delta": {"content": "The first half of the answer"}}]}))
        self.wfile.flush()
        time.sleep(0.05)
        self.wfile.write(sse({"id": "chatcmpl-fail",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": {"message": "MTPLX ran out of memory while generating.",
                      "type": "server_error", "code": "insufficient_memory", "param": None}}))
        self.wfile.write(b"data: [DONE]\\n\\n")
        """)
        defer { daemon.terminate() }
        let (viewModel, container) = try makeChatViewModel(daemon: daemon)

        viewModel.send("hello")
        let conversation = try XCTUnwrap(viewModel.current)
        try await pollUntil("the failed turn settles") { !viewModel.isStreaming && viewModel.lastError != nil }

        // The user sees the daemon's own message and can retry.
        XCTAssertEqual(viewModel.lastError, .server("MTPLX ran out of memory while generating."))
        XCTAssertEqual(
            viewModel.lastError?.errorDescription,
            "Reply failed: MTPLX ran out of memory while generating."
        )
        XCTAssertTrue(viewModel.canRetryLastUserMessage)

        // The partial text is kept, the turn is a failure (not a
        // completion), and the message rides with it for the transcript.
        let persisted = try persistedAssistantMessages(in: container, conversationID: conversation.id)
        XCTAssertEqual(persisted.count, 1)
        let message = try XCTUnwrap(persisted.first)
        XCTAssertEqual(message.visibleContent, "The first half of the answer")
        XCTAssertEqual(message.finishReason, "error")
        XCTAssertEqual(
            ChatTurnFailure.decode(fromStatsJSON: message.statsJSON),
            ChatTurnFailure(errorMessage: "MTPLX ran out of memory while generating.")
        )
        XCTAssertEqual(viewModel.visibleMessages.map(\.role), [.user, .assistant])
        XCTAssertFalse(daemon.cancelWasCalled)
    }

    @MainActor
    func testOrdinaryStopStillCompletesWithoutError() async throws {
        let daemon = try await ChatFakeDaemon.start(chatHandler: """
        self.wfile.write(sse({"id": "chatcmpl-ok",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
        self.wfile.write(sse({"id": "chatcmpl-ok",
            "choices": [{"index": 0, "delta": {"content": "Whole answer."}}]}))
        self.wfile.write(sse({"id": "chatcmpl-ok",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}))
        self.wfile.write(b"data: [DONE]\\n\\n")
        """)
        defer { daemon.terminate() }
        let (viewModel, container) = try makeChatViewModel(daemon: daemon)

        viewModel.send("hello")
        let conversation = try XCTUnwrap(viewModel.current)
        try await pollUntil("the turn settles") {
            !viewModel.isStreaming && viewModel.visibleMessages.last?.role == .assistant
        }

        XCTAssertNil(viewModel.lastError)
        let persisted = try persistedAssistantMessages(in: container, conversationID: conversation.id)
        XCTAssertEqual(persisted.map(\.visibleContent), ["Whole answer."])
        XCTAssertEqual(persisted.first?.finishReason, "stop")
        XCTAssertNil(ChatTurnFailure.decode(fromStatsJSON: persisted.first?.statsJSON))
    }

    // MARK: Persisted failure sidecar

    func testFailureRidesBesideStatsInOneBlobAndBothDecode() throws {
        let stats = ChatTurnStats(rawDecodeTokS: 36.5, completionTokens: 12, thinkingTimeMs: 640)
        let failure = ChatTurnFailure(errorMessage: "boom")
        let json = try XCTUnwrap(ChatTurnFailure.statsJSON(stats: stats, failure: failure))

        let decodedStats = try JSONDecoder().decode(ChatTurnStats.self, from: Data(json.utf8))
        XCTAssertEqual(decodedStats.rawDecodeTokS, 36.5)
        XCTAssertEqual(decodedStats.completionTokens, 12)
        XCTAssertEqual(decodedStats.thinkingTimeMs, 640)
        XCTAssertEqual(ChatTurnFailure.decode(fromStatsJSON: json), failure)

        // A stats-only blob (every turn persisted before this fix) has no failure.
        let statsOnly = try XCTUnwrap(ChatTurnFailure.statsJSON(stats: stats, failure: nil))
        XCTAssertNil(ChatTurnFailure.decode(fromStatsJSON: statsOnly))
        XCTAssertNil(ChatTurnFailure.decode(fromStatsJSON: nil))
        XCTAssertNil(ChatTurnFailure.statsJSON(stats: nil, failure: nil))
    }
}
