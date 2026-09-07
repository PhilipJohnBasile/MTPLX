import Foundation
import XCTest

@testable import MTPLXAppCore

// MARK: - StreamLostTests
//
// A reply the daemon never finished — the process died or the
// connection was cut mid-generation — ends the client's byte stream
// with no terminal chunk. URLSession reports that as a normal end for
// a clean close AND for an HTTP/1.1 chunked body cut before its
// terminating chunk (both reproduced here against the scripted
// daemon). The round's finish reason used to default to "stop", so the
// half answer was persisted as a complete reply with no label, no error
// card and no Retry. Now the absence of the finish frame is the signal:
// the partial is kept, filed as incomplete, and the user is offered
// Retry. Normal completion and the user's own Stop are unchanged.

final class StreamLostTests: XCTestCase {

    private static let halfAnswer = "Half an answer and then"

    @MainActor
    func testCleanCloseWithoutTerminalChunkIsStreamLostWithRetry() async throws {
        let daemon = try await ChatFakeDaemon.start(chatHandler: """
        self.wfile.write(sse({"id": "chatcmpl-lost",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
        self.wfile.write(sse({"id": "chatcmpl-lost",
            "choices": [{"index": 0, "delta": {"content": "\(Self.halfAnswer)"}}]}))
        self.wfile.flush()
        time.sleep(0.1)
        # Return with no finish chunk and no [DONE]: the HTTP/1.0
        # response ends in a clean socket close.
        """)
        defer { daemon.terminate() }
        let (viewModel, container) = try makeChatViewModel(daemon: daemon)

        viewModel.send("hello")
        let conversation = try XCTUnwrap(viewModel.current)
        try await pollUntil("the lost turn settles") { !viewModel.isStreaming }

        XCTAssertEqual(viewModel.lastError, .streamLost)
        XCTAssertTrue(viewModel.canRetryLastUserMessage)
        let persisted = try persistedAssistantMessages(in: container, conversationID: conversation.id)
        XCTAssertEqual(persisted.count, 1)
        XCTAssertEqual(persisted.first?.visibleContent, Self.halfAnswer, "the partial is kept")
        XCTAssertEqual(persisted.first?.finishReason, ChatViewModel.streamLostFinishReason)
        XCTAssertEqual(ChatViewModel.streamLostFinishReason, "incomplete")
        XCTAssertFalse(daemon.cancelWasCalled)
    }

    @MainActor
    func testChunkedBodyCutBeforeItsTerminatingChunkIsStreamLost() async throws {
        // The framing a real ASGI daemon uses; the socket is dropped
        // before the zero-length chunk, as when the process dies.
        let daemon = try await ChatFakeDaemon.start(chatHandler: """
        self.wfile.write(self.frame({"id": "chatcmpl-crash",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
        self.wfile.write(self.frame({"id": "chatcmpl-crash",
            "choices": [{"index": 0, "delta": {"content": "\(Self.halfAnswer)"}}]}))
        self.wfile.flush()
        time.sleep(0.1)
        self.drop()
        """, chunked: true)
        defer { daemon.terminate() }
        let (viewModel, container) = try makeChatViewModel(daemon: daemon)

        viewModel.send("hello")
        let conversation = try XCTUnwrap(viewModel.current)
        try await pollUntil("the cut turn settles") { !viewModel.isStreaming }

        XCTAssertEqual(viewModel.lastError, .streamLost)
        XCTAssertTrue(viewModel.canRetryLastUserMessage)
        let persisted = try persistedAssistantMessages(in: container, conversationID: conversation.id)
        XCTAssertEqual(persisted.map(\.visibleContent), [Self.halfAnswer])
        XCTAssertEqual(persisted.first?.finishReason, "incomplete")
    }

    @MainActor
    func testLostStreamWithNoTextLeavesRetryAndNoPhantomReply() async throws {
        let daemon = try await ChatFakeDaemon.start(chatHandler: """
        self.wfile.write(sse({"id": "chatcmpl-lost",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
        self.wfile.flush()
        time.sleep(0.05)
        """)
        defer { daemon.terminate() }
        let (viewModel, container) = try makeChatViewModel(daemon: daemon)

        viewModel.send("hello")
        let conversation = try XCTUnwrap(viewModel.current)
        try await pollUntil("the lost turn settles") { !viewModel.isStreaming }

        XCTAssertEqual(viewModel.lastError, .streamLost)
        XCTAssertTrue(viewModel.canRetryLastUserMessage)
        // Nothing arrived, so there is nothing to file — same as a
        // transport error before the first token.
        XCTAssertTrue(try persistedAssistantMessages(in: container, conversationID: conversation.id).isEmpty)
        XCTAssertEqual(viewModel.visibleMessages.map(\.role), [.user])
    }

    @MainActor
    func testChunkedStreamWithTerminalChunkCompletesAsBefore() async throws {
        let daemon = try await ChatFakeDaemon.start(chatHandler: """
        self.wfile.write(self.frame({"id": "chatcmpl-ok",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
        self.wfile.write(self.frame({"id": "chatcmpl-ok",
            "choices": [{"index": 0, "delta": {"content": "Whole answer."}}]}))
        self.wfile.write(self.frame({"id": "chatcmpl-ok",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}))
        self.done()
        """, chunked: true)
        defer { daemon.terminate() }
        let (viewModel, container) = try makeChatViewModel(daemon: daemon)

        viewModel.send("hello")
        let conversation = try XCTUnwrap(viewModel.current)
        try await pollUntil("the turn settles") {
            !viewModel.isStreaming && viewModel.visibleMessages.last?.role == .assistant
        }

        XCTAssertNil(viewModel.lastError)
        XCTAssertFalse(viewModel.canRetryLastUserMessage)
        let persisted = try persistedAssistantMessages(in: container, conversationID: conversation.id)
        XCTAssertEqual(persisted.map(\.visibleContent), ["Whole answer."])
        XCTAssertEqual(persisted.first?.finishReason, "stop")
    }

    @MainActor
    func testLengthFinishIsStillACompletion() async throws {
        let daemon = try await ChatFakeDaemon.start(chatHandler: """
        self.wfile.write(sse({"id": "chatcmpl-len",
            "choices": [{"index": 0, "delta": {"content": "Cut by the token limit"}}]}))
        self.wfile.write(sse({"id": "chatcmpl-len",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}))
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
        XCTAssertEqual(persisted.first?.finishReason, "length")
    }

    @MainActor
    func testUserStopIsUnchanged() async throws {
        let daemon = try await ChatFakeDaemon.start(chatHandler: """
        self.wfile.write(sse({"id": "chatcmpl-hold",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
        self.wfile.write(sse({"id": "chatcmpl-hold",
            "choices": [{"index": 0, "delta": {"content": "partial answer"}}]}))
        self.wfile.flush()
        time.sleep(3)
        """)
        defer { daemon.terminate() }
        let (viewModel, container) = try makeChatViewModel(daemon: daemon)

        viewModel.send("start a long answer")
        let conversation = try XCTUnwrap(viewModel.current)
        try await pollUntil("the first fragment arrives") { viewModel.hasStreamingContent }

        await viewModel.cancel()
        try await pollUntil("the cancelled turn settles") { !viewModel.isStreaming }

        XCTAssertNil(viewModel.lastError, "a user stop is not an error")
        let persisted = try persistedAssistantMessages(in: container, conversationID: conversation.id)
        XCTAssertEqual(persisted.map(\.visibleContent), ["partial answer"])
        XCTAssertEqual(persisted.first?.finishReason, "cancelled")
        XCTAssertTrue(daemon.cancelWasCalled)
    }
}
