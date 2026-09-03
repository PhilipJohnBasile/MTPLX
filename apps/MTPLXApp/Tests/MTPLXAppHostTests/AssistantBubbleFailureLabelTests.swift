import XCTest

@testable import MTPLXAppCore
@testable import MTPLXAppHost

/// The settled bubble's label for a turn the daemon failed: the server's
/// own message, verbatim, behind a localised "Failed:" prefix. A user
/// stop or a failure recorded before the message was persisted keeps
/// the plain "Interrupted reply" caption.
final class AssistantBubbleFailureLabelTests: XCTestCase {
    override func setUp() {
        super.setUp()
        L10n.activate(.english)
    }

    override func tearDown() {
        L10n.activate(.english)
        super.tearDown()
    }

    func testErrorFinishWithPersistedMessageReadsAsFailedWithTheServerMessage() {
        let statsJSON = ChatTurnFailure.statsJSON(
            stats: nil,
            failure: ChatTurnFailure(errorMessage: "MTPLX ran out of memory while generating.")
        )
        let failure = AssistantBubbleView.failure(finishReason: "error", statsJSON: statsJSON)
        XCTAssertEqual(failure?.errorMessage, "MTPLX ran out of memory while generating.")
        XCTAssertEqual(
            AssistantBubbleView.interruptedReplyTitle(failure: failure),
            "Failed: MTPLX ran out of memory while generating."
        )
    }

    func testFailedLabelIsLocalisedAroundTheVerbatimServerMessage() {
        L10n.activate(.german)
        let title = AssistantBubbleView.interruptedReplyTitle(
            failure: ChatTurnFailure(errorMessage: "context window exceeded")
        )
        XCTAssertEqual(title, "Fehlgeschlagen: context window exceeded")
    }

    func testCancelledAndLegacyErrorTurnsKeepInterruptedReply() {
        // A user stop never carries a failure, whatever the blob holds.
        let blob = ChatTurnFailure.statsJSON(stats: nil, failure: ChatTurnFailure(errorMessage: "x"))
        XCTAssertNil(AssistantBubbleView.failure(finishReason: "cancelled", statsJSON: blob))
        // An error persisted before the message existed has stats only.
        let statsOnly = ChatTurnFailure.statsJSON(stats: ChatTurnStats(rawDecodeTokS: 30), failure: nil)
        XCTAssertNil(AssistantBubbleView.failure(finishReason: "error", statsJSON: statsOnly))
        XCTAssertNil(AssistantBubbleView.failure(finishReason: "error", statsJSON: nil))
        XCTAssertEqual(AssistantBubbleView.interruptedReplyTitle(failure: nil), "Interrupted reply")
    }
}
