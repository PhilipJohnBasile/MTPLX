import XCTest
@testable import MTPLXAppCore

/// The activity strip's search chip counts only calls that ran; a failed
/// search or fetch is labelled as a failure, never as "Searched" or a
/// page read. Covers the live model (pending-trace statuses) and the
/// settled model (persisted trace statuses), plus the grouped turn's
/// success/failure split that feeds the settled bubble.
final class ToolFailureActivityTests: XCTestCase {
    override func setUp() {
        super.setUp()
        L10n.activate(.english)
    }

    // MARK: Live strip

    func testLiveFailedSearchReadsAsSearchFailed() {
        let traces = [
            PendingToolTrace(id: "r1-c1", name: "web_search", detail: "Search failed: offline", status: .failed)
        ]
        let model = TurnActivityModel.live(phase: .thinking, hasReasoning: true, traces: traces)
        XCTAssertEqual(model.chips.map(\.kind), [.thought, .search])
        XCTAssertEqual(model.chips[1].label, "Search failed")
        XCTAssertNil(model.chips[1].caption)
        XCTAssertFalse(model.chips[1].isLive)
    }

    func testLiveMixedOutcomesCountSuccessesAndNameFailures() {
        let traces = [
            PendingToolTrace(id: "r1-c1", name: "web_search", status: .success),
            PendingToolTrace(id: "r1-c2", name: "web_search", status: .success),
            PendingToolTrace(id: "r1-c3", name: "web_search", status: .failed),
        ]
        let model = TurnActivityModel.live(phase: .answering, hasReasoning: false, traces: traces)
        XCTAssertEqual(model.chips.map(\.label), ["Searched"])
        XCTAssertEqual(model.chips[0].caption, "×2 · 1 failed")
    }

    func testLivePendingSearchIsStillLiveWhateverEarlierCallsDid() {
        let traces = [
            PendingToolTrace(id: "r1-c1", name: "web_search", status: .failed),
            PendingToolTrace(id: "r1-c2", name: "web_search", status: .pending),
        ]
        let model = TurnActivityModel.live(phase: .searching, hasReasoning: false, traces: traces)
        XCTAssertEqual(model.chips[0].label, "Searching")
        XCTAssertTrue(model.chips[0].isLive)
    }

    func testLiveFailedFetchReadsAsFetchFailed() {
        let traces = [
            PendingToolTrace(id: "r1-c1", name: "fetch_url", status: .failed),
            PendingToolTrace(id: "r1-c2", name: "fetch_url", status: .failed),
        ]
        let model = TurnActivityModel.live(phase: .answering, hasReasoning: false, traces: traces)
        XCTAssertEqual(model.chips.map(\.label), ["Fetch failed"])
        XCTAssertEqual(model.chips[0].caption, "×2")
    }

    // MARK: Settled strip

    func testSettledAllSearchesFailedNamesTheFailure() {
        let model = TurnActivityModel.settled(
            hasThought: false,
            thinkingTimeMs: nil,
            searchCount: 0,
            fetchedPageCount: 0,
            hasOtherToolActivity: true,
            failedSearchCount: 1,
            failedFetchCount: 0
        )
        XCTAssertEqual(model.chips.map(\.label), ["Search failed"])
        XCTAssertNil(model.chips[0].caption)
    }

    func testSettledSuccessesKeepTheirLabelAndCountFailuresSeparately() {
        let model = TurnActivityModel.settled(
            hasThought: true,
            thinkingTimeMs: 1_200,
            searchCount: 1,
            fetchedPageCount: 0,
            hasOtherToolActivity: true,
            failedSearchCount: 0,
            failedFetchCount: 2
        )
        XCTAssertEqual(model.chips.map(\.label), ["Thought", "Searched"])
        XCTAssertEqual(model.chips[1].caption, "2 failed")
    }

    func testSettledFailedFetchesOnlyReadAsFetchFailed() {
        let model = TurnActivityModel.settled(
            hasThought: false,
            thinkingTimeMs: nil,
            searchCount: 0,
            fetchedPageCount: 0,
            hasOtherToolActivity: true,
            failedSearchCount: 0,
            failedFetchCount: 1
        )
        XCTAssertEqual(model.chips.map(\.label), ["Fetch failed"])
    }

    func testSettledLabelsAreLocalised() {
        L10n.activate(.german)
        defer { L10n.activate(.english) }
        let model = TurnActivityModel.settled(
            hasThought: false,
            thinkingTimeMs: nil,
            searchCount: 0,
            fetchedPageCount: 0,
            hasOtherToolActivity: true,
            failedSearchCount: 2,
            failedFetchCount: 0
        )
        XCTAssertEqual(model.chips[0].label, "Suche fehlgeschlagen")
        XCTAssertEqual(model.chips[0].caption, "×2")
    }

    // MARK: Grouped turn

    func testGroupSplitsSearchesAndFetchesBySuccess() {
        let turnID = UUID()
        let assistant = ChatMessage(
            role: .assistant, visibleContent: "", finishReason: "tool_calls", turnGroupID: turnID
        )
        let ok = ToolTraceRecord(
            name: "web_search", status: .success,
            argumentsJSON: #"{"query":"mtplx"}"#,
            resultJSON: #"{"query":"mtplx","results":[{"title":"MTPLX","url":"https://example.com/a"}]}"#,
            startedAt: Date(timeIntervalSince1970: 1), message: assistant
        )
        let failed = ToolTraceRecord(
            name: "web_search", status: .failed,
            argumentsJSON: #"{"query":"mtplx benchmarks"}"#,
            resultJSON: #"{"error":"search_failed","detail":"offline","query":"mtplx benchmarks"}"#,
            startedAt: Date(timeIntervalSince1970: 2), message: assistant
        )
        let fetchFailed = ToolTraceRecord(
            name: "fetch_url", status: .failed,
            argumentsJSON: #"{"url":"https://example.com/b"}"#,
            resultJSON: #"{"error":"fetch_failed","detail":"HTTP 503","url":"https://example.com/b"}"#,
            startedAt: Date(timeIntervalSince1970: 3), message: assistant
        )
        let fetched = ToolTraceRecord(
            name: "fetch_url", status: .success,
            argumentsJSON: #"{"url":"https://example.com/c"}"#,
            resultJSON: #"{"url":"https://example.com/c","title":"C","content":"..."}"#,
            startedAt: Date(timeIntervalSince1970: 4), message: assistant
        )
        assistant.toolTraces = [ok, failed, fetchFailed, fetched]
        let answer = ChatMessage(
            role: .assistant, visibleContent: "answer", finishReason: "stop", turnGroupID: turnID
        )
        let group = AssistantTurnGroup(id: turnID, members: [assistant, answer])

        XCTAssertEqual(group.successfulSearchCount, 1)
        XCTAssertEqual(group.failedSearchCount, 1)
        XCTAssertEqual(group.fetchedPageCount, 1)
        XCTAssertEqual(group.failedFetchCount, 1)
        XCTAssertEqual(group.searchQueries, ["mtplx", "mtplx benchmarks"])
        XCTAssertEqual(
            group.searchReceipts,
            [
                AssistantTurnGroup.SearchReceipt(query: "mtplx", failureDetail: nil),
                AssistantTurnGroup.SearchReceipt(query: "mtplx benchmarks", failureDetail: "offline"),
            ]
        )
        // A failed search contributes no source; the footer lists only
        // what was actually read.
        XCTAssertEqual(group.sources.map(\.url), ["https://example.com/a", "https://example.com/c"])
    }

    func testGroupTreatsRowsPersistedBeforeFailureRecordingAsSuccesses() {
        // Every trace persisted before this fix carries `.success`; an
        // old turn keeps reading "Searched ×2".
        let turnID = UUID()
        let assistant = ChatMessage(role: .assistant, visibleContent: "", turnGroupID: turnID)
        assistant.toolTraces = [
            ToolTraceRecord(name: "web_search", status: .success, argumentsJSON: #"{"query":"a"}"#, message: assistant),
            ToolTraceRecord(name: "web_search", status: .success, argumentsJSON: #"{"query":"b"}"#, message: assistant),
        ]
        let group = AssistantTurnGroup(id: turnID, members: [assistant])
        XCTAssertEqual(group.successfulSearchCount, 2)
        XCTAssertEqual(group.failedSearchCount, 0)
        let model = TurnActivityModel.settled(
            hasThought: false,
            thinkingTimeMs: nil,
            searchCount: group.successfulSearchCount,
            fetchedPageCount: group.fetchedPageCount,
            hasOtherToolActivity: true,
            failedSearchCount: group.failedSearchCount,
            failedFetchCount: group.failedFetchCount
        )
        XCTAssertEqual(model.chips[0].label, "Searched")
        XCTAssertEqual(model.chips[0].caption, "×2")
    }
}
