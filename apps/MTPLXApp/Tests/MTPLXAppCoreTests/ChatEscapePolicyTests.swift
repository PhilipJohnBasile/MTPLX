import XCTest
@testable import MTPLXAppCore

/// Esc on the chat surface is one decision: stop a streaming reply,
/// otherwise close the surface. The surface's single Esc binding and
/// nothing else consult it, so the outcome no longer depends on which
/// of two bindings the responder chain reached first.
final class ChatEscapePolicyTests: XCTestCase {
    func testEscapeStopsGenerationWhileAReplyStreams() {
        XCTAssertEqual(ChatEscapePolicy.action(isStreaming: true), .stopGenerating)
    }

    func testEscapeClosesTheSurfaceWhenIdle() {
        XCTAssertEqual(ChatEscapePolicy.action(isStreaming: false), .closeSurface)
    }

    func testTheTwoOutcomesAreDistinct() {
        XCTAssertNotEqual(
            ChatEscapePolicy.action(isStreaming: true),
            ChatEscapePolicy.action(isStreaming: false)
        )
    }
}
