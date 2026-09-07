import Foundation
import SwiftData
import XCTest
@testable import MTPLXAppCore

// Issue #349: "No response from any tools after install." A fresh-install
// user asked the built-in chat about a project in ~/Dev; the model invoked
// tools (`ls`, `find`, `search_files`, `read_file`, `echo`) and received
// NOTHING back — no output, no error. These tests pin the app-side
// truthfulness invariants:
//
//   1. Any tool call the app cannot execute produces a NON-EMPTY,
//      explanatory result string (never a blank the model reads as "the
//      call went into the void").
//   2. That result survives request-payload compaction and reaches the
//      conversation payload for the next turn.
//   3. The Settings "Agent workspace" folder actually reaches the hermes
//      subprocess surfaces (launch environment + terminal.cwd config) —
//      the working-folder plumbing the reporter assumed was broken.
final class ToolResultIntegrityTests: XCTestCase {

    // MARK: - 1. Unknown / unexecuted tools never yield a blank result

    func testUnknownToolDispatchReturnsNonEmptyTruthfulResult() async {
        let factory = MTPLXChatToolFactory()
        for name in ["terminal", "search_files", "read_file", "ls", "echo"] {
            let outcome = await factory.dispatch(
                name: name,
                argumentsJSON: #"{"command":"ls ~/Dev"}"#
            )
            let result = outcome.resultJSON
            XCTAssertFalse(
                result.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "dispatch(\(name)) returned a blank tool result"
            )
            XCTAssertTrue(result.contains("unknown_tool"), result)
            XCTAssertTrue(result.contains(name), result)
            XCTAssertEqual(outcome.failure?.kind, .unknownTool)
        }
    }

    func testFetchURLWithInvalidURLReturnsNonEmptyError() async {
        let factory = MTPLXChatToolFactory()
        let outcome = await factory.dispatch(
            name: "fetch_url",
            argumentsJSON: #"{"url":"not a url"}"#
        )
        XCTAssertFalse(outcome.resultJSON.isEmpty)
        XCTAssertTrue(outcome.resultJSON.contains("invalid_url"), outcome.resultJSON)
        XCTAssertEqual(outcome.failure?.kind, .invalidURL)
        XCTAssertFalse(outcome.succeeded)
    }

    // MARK: - 1b. Failures are typed, and the model is told the tool failed (B-05)
    //
    // The factory used to answer every failure with a JSON blob that the
    // dispatch loop recorded as a success, so an offline search showed
    // "Searched" in the activity strip and the model, handed
    // `{"results": []}` or `{"error": "search_failed"}` as a result,
    // answered that nothing was found.

    /// A transport with no network: every request throws.
    private struct OfflineWebTransport: WebTransport {
        func data(for request: URLRequest) async throws -> (Data, URLResponse) {
            throw URLError(.notConnectedToInternet)
        }
    }

    /// A transport that answers every request with the same body.
    private struct FixtureWebTransport: WebTransport {
        let body: String
        func data(for request: URLRequest) async throws -> (Data, URLResponse) {
            let response = HTTPURLResponse(
                url: request.url ?? URL(string: "https://example.com")!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "text/html"]
            )!
            return (Data(body.utf8), response)
        }
    }

    private func offlineFactory() -> MTPLXChatToolFactory {
        MTPLXChatToolFactory(
            webSearch: WebSearchService(transport: OfflineWebTransport(), cache: WebSearchCache()),
            urlFetcher: URLFetcher(transport: OfflineWebTransport(), cache: URLFetchCache())
        )
    }

    func testWebSearchWithNoReachableProviderIsASearchFailureNotAnEmptyResult() async throws {
        let outcome = await offlineFactory().dispatch(
            name: "web_search",
            argumentsJSON: #"{"query":"mtplx release notes"}"#
        )
        let failure = try XCTUnwrap(outcome.failure, "an offline search must be a failure, not 'no results'")
        XCTAssertEqual(failure.kind, .searchFailed)
        XCTAssertFalse(failure.detail.isEmpty)

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(outcome.resultJSON.utf8)) as? [String: Any]
        )
        XCTAssertEqual(object["error"] as? String, "search_failed")
        XCTAssertEqual(object["query"] as? String, "mtplx release notes")
        XCTAssertNil(object["results"], "a failed search must not look like an empty result set")
        let note = try XCTUnwrap(object["note"] as? String)
        XCTAssertTrue(note.localizedCaseInsensitiveContains("failed"), note)
        XCTAssertTrue(note.contains("do not claim that nothing was found"), note)
    }

    func testWebSearchStillSucceedsWithEmptyResultsWhenProvidersAnswer() async throws {
        // Providers reachable but returning a page with no results: the
        // search ran and found nothing, which is a success, not a failure.
        let factory = MTPLXChatToolFactory(
            webSearch: WebSearchService(
                transport: FixtureWebTransport(body: "<html><body>no results here</body></html>"),
                cache: WebSearchCache()
            )
        )
        let outcome = await factory.dispatch(
            name: "web_search",
            argumentsJSON: #"{"query":"something obscure"}"#
        )
        XCTAssertTrue(outcome.succeeded)
        XCTAssertTrue(outcome.resultJSON.contains("\"results\":[]"), outcome.resultJSON)
    }

    func testFetchURLWithNoNetworkIsAFetchFailure() async throws {
        let outcome = await offlineFactory().dispatch(
            name: "fetch_url",
            argumentsJSON: #"{"url":"https://example.com/release"}"#
        )
        XCTAssertEqual(outcome.failure?.kind, .fetchFailed)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(outcome.resultJSON.utf8)) as? [String: Any]
        )
        XCTAssertEqual(object["error"] as? String, "fetch_failed")
        XCTAssertEqual(object["url"] as? String, "https://example.com/release")
        XCTAssertFalse((object["detail"] as? String ?? "").isEmpty)
        XCTAssertTrue((object["note"] as? String ?? "").contains("could not be fetched"))
    }

    func testEmptyQueryIsATypedFailureThatStillNamesTheProblemToTheModel() async {
        let outcome = await MTPLXChatToolFactory().dispatch(
            name: "web_search",
            argumentsJSON: #"{"query":"   "}"#
        )
        XCTAssertEqual(outcome.failure?.kind, .emptyQuery)
        XCTAssertTrue(outcome.resultJSON.contains("empty_query"), outcome.resultJSON)
    }

    @MainActor
    func testFailureDetailLabelsTheKindAndKeepsTheReason() {
        L10n.activate(.english)
        XCTAssertEqual(
            ChatViewModel.failureDetail(ChatToolFailure(kind: .searchFailed, detail: "offline")),
            "Search failed: offline"
        )
        XCTAssertEqual(
            ChatViewModel.failureDetail(ChatToolFailure(kind: .fetchFailed, detail: "HTTP 503")),
            "Fetch failed: HTTP 503"
        )
        XCTAssertEqual(
            ChatViewModel.failureDetail(ChatToolFailure(kind: .unknownTool, detail: "ls is not a tool")),
            "Tool failed: ls is not a tool"
        )
    }

    /// End to end: the model calls web_search, the network is down, and the
    /// turn records a FAILED trace whose result tells the model so.
    @MainActor
    func testFailedWebSearchIsRecordedAsFailedAndReachesTheModelAsAFailure() async throws {
        let daemon = try await ChatFakeDaemon.start(chatHandler: """
        if count == 1:
            self.wfile.write(sse({"id": "chatcmpl-web",
                "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
            self.wfile.write(sse({"id": "chatcmpl-web",
                "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_search",
                    "type": "function", "function": {"name": "web_search", "arguments": ""}}]}}]}))
            self.wfile.write(sse({"id": "chatcmpl-web",
                "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0,
                    "function": {"arguments": json.dumps({"query": "latest mtplx release"})}}]}}]}))
            self.wfile.write(sse({"id": "chatcmpl-web",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}))
        else:
            self.wfile.write(sse({"id": "chatcmpl-web-final",
                "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
            self.wfile.write(sse({"id": "chatcmpl-web-final",
                "choices": [{"index": 0, "delta": {"content": "The search failed, so from memory: ..."}}]}))
            self.wfile.write(sse({"id": "chatcmpl-web-final",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
        self.wfile.write(b"data: [DONE]\\n\\n")
        """)
        defer { daemon.terminate() }
        let (viewModel, container) = try makeChatViewModel(daemon: daemon, toolFactory: offlineFactory())
        _ = viewModel.createNewConversation()
        viewModel.webSearchEnabled = true

        viewModel.send("What is new in the latest release?")
        let conversation = try XCTUnwrap(viewModel.current)
        try await pollUntil("the two-round turn settles") {
            !viewModel.isStreaming && viewModel.visibleMessages.last?.role == .assistant
        }
        XCTAssertNil(viewModel.lastError)

        // The persisted trace is a failure with the reason attached.
        let assistantMessages = try persistedAssistantMessages(in: container, conversationID: conversation.id)
        let traces = assistantMessages.flatMap(\.toolTraces)
        XCTAssertEqual(traces.count, 1)
        let trace = try XCTUnwrap(traces.first)
        XCTAssertEqual(trace.name, "web_search")
        XCTAssertEqual(trace.status, .failed)
        XCTAssertTrue((trace.resultJSON ?? "").contains("search_failed"), trace.resultJSON ?? "nil")

        // The grouped turn counts it as a failure, not a search that ran,
        // and the settled chip says so.
        let group = AssistantTurnGroup(id: conversation.id, members: assistantMessages)
        XCTAssertEqual(group.failedSearchCount, 1)
        XCTAssertEqual(group.successfulSearchCount, 0)
        XCTAssertEqual(group.searchReceipts.map(\.query), ["latest mtplx release"])
        XCTAssertTrue(group.searchReceipts[0].failed)
        let chips = TurnActivityModel.settled(
            hasThought: false,
            thinkingTimeMs: nil,
            searchCount: group.successfulSearchCount,
            fetchedPageCount: group.fetchedPageCount,
            hasOtherToolActivity: !group.traces.isEmpty,
            failedSearchCount: group.failedSearchCount,
            failedFetchCount: group.failedFetchCount
        ).chips
        XCTAssertEqual(chips.map(\.label), ["Search failed"])

        // The tool message the model received in round two names the failure.
        let toolMessages = try persistedToolMessages(in: container, conversationID: conversation.id)
        XCTAssertEqual(toolMessages.count, 1)
        let toolResult = try XCTUnwrap(toolMessages.first?.visibleContent)
        XCTAssertTrue(toolResult.contains("\"error\":\"search_failed\""), toolResult)
        XCTAssertTrue(toolResult.contains("Tell the user the search failed"), toolResult)
        XCTAssertFalse(toolResult.contains("\"results\""), "a failed search must not look like an empty result set")
    }

    /// A failed fetch names its URL in the call arguments; it must not
    /// turn into a source pill for a page that was never read.
    @MainActor
    func testFailedFetchIsNotCountedAsASource() async throws {
        let daemon = try await ChatFakeDaemon.start(chatHandler: """
        if count == 1:
            self.wfile.write(sse({"id": "chatcmpl-fetch",
                "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_fetch",
                    "type": "function", "function": {"name": "fetch_url",
                    "arguments": json.dumps({"url": "https://example.com/release"})}}]}}]}))
            self.wfile.write(sse({"id": "chatcmpl-fetch",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}))
        else:
            self.wfile.write(sse({"id": "chatcmpl-fetch-final",
                "choices": [{"index": 0, "delta": {"content": "I could not open that page."}}]}))
            self.wfile.write(sse({"id": "chatcmpl-fetch-final",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
        self.wfile.write(b"data: [DONE]\\n\\n")
        """)
        defer { daemon.terminate() }
        let (viewModel, container) = try makeChatViewModel(daemon: daemon, toolFactory: offlineFactory())
        _ = viewModel.createNewConversation()
        viewModel.webSearchEnabled = true

        viewModel.send("Read https://example.com/release")
        let conversation = try XCTUnwrap(viewModel.current)
        try await pollUntil("the two-round turn settles") {
            !viewModel.isStreaming && viewModel.visibleMessages.last?.role == .assistant
        }

        let assistantMessages = try persistedAssistantMessages(in: container, conversationID: conversation.id)
        let group = AssistantTurnGroup(id: conversation.id, members: assistantMessages)
        XCTAssertEqual(group.failedFetchCount, 1)
        XCTAssertEqual(group.fetchedPageCount, 0)
        XCTAssertTrue(group.sources.isEmpty, "a page that was never read is not a source")
        XCTAssertNil(group.finalMessage.sourcesJSON)
        XCTAssertTrue(viewModel.liveTurnSources.isEmpty)
    }

    @MainActor
    private func persistedToolMessages(
        in container: ModelContainer,
        conversationID: UUID
    ) throws -> [ChatMessage] {
        let descriptor = FetchDescriptor<ChatMessage>(
            predicate: #Predicate<ChatMessage> { message in
                message.conversationID == conversationID
            },
            sortBy: [SortDescriptor(\.createdAt)]
        )
        return try container.mainContext.fetch(descriptor).filter { $0.role == .tool }
    }

    @MainActor
    func testUnexecutedToolResultJSONIsNonEmptyAndNamesTheTool() throws {
        let json = ChatViewModel.unexecutedToolResultJSON(toolName: "terminal")
        XCTAssertTrue(json.contains("tool_not_executed"), json)
        XCTAssertTrue(json.contains("terminal"), json)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        )
        let note = try XCTUnwrap(object["note"] as? String)
        XCTAssertFalse(note.isEmpty)
        XCTAssertTrue(note.contains("did not execute"))

        // A nameless call still yields a truthful, non-empty result.
        let nameless = ChatViewModel.unexecutedToolResultJSON(toolName: "")
        XCTAssertFalse(nameless.isEmpty)
        XCTAssertTrue(nameless.contains("tool_not_executed"), nameless)
    }

    // MARK: - 2. Results survive compaction and reach the request payload

    @MainActor
    func testCompactToolResultContentNeverEmptiesErrorResults() {
        let short = ChatViewModel.unexecutedToolResultJSON(toolName: "read_file")
        XCTAssertEqual(ChatViewModel.compactToolResultContent(short), short)

        let overLimit =
            "{\"error\":\"tool_not_executed\",\"content\":\""
            + String(repeating: "a", count: ChatViewModel.requestToolResultContentLimit + 8_000)
            + "\"}"
        let compacted = ChatViewModel.compactToolResultContent(overLimit)
        XCTAssertFalse(
            compacted.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "compaction turned an oversized tool result into a blank"
        )

        let overLimitNonJSON = String(
            repeating: "x",
            count: ChatViewModel.requestToolResultContentLimit + 5_000
        )
        let clamped = ChatViewModel.compactToolResultContent(overLimitNonJSON)
        XCTAssertFalse(clamped.isEmpty)
    }

    @MainActor
    func testBuildRequestMessagesCarriesNonEmptyToolResultForEveryCall() throws {
        let conversation = ChatConversation(title: "Dangling repair")
        let user = ChatMessage(
            role: .user,
            visibleContent: "look at ~/Dev",
            createdAt: Date(timeIntervalSince1970: 100),
            conversation: conversation
        )
        let records = [
            ToolCallRecord(
                id: "call_term",
                name: "terminal",
                arguments: #"{"command":"ls ~/Dev"}"#
            )
        ]
        let toolCallsJSON = try XCTUnwrap(
            String(data: JSONEncoder().encode(records), encoding: .utf8)
        )
        let assistant = ChatMessage(
            role: .assistant,
            visibleContent: "",
            toolCallsJSON: toolCallsJSON,
            finishReason: "tool_calls",
            createdAt: Date(timeIntervalSince1970: 101),
            conversation: conversation
        )
        let toolResult = ChatMessage(
            role: .tool,
            visibleContent: ChatViewModel.unexecutedToolResultJSON(toolName: "terminal"),
            toolCallId: "call_term",
            createdAt: Date(timeIntervalSince1970: 102),
            conversation: conversation
        )

        let request = ChatViewModel.buildRequestMessages(
            from: [user, assistant, toolResult],
            overrideLastUserContent: nil
        )

        let assistantEntry = try XCTUnwrap(
            request.first { $0.role == "assistant" }
        )
        XCTAssertEqual(assistantEntry.toolCalls?.first?.id, "call_term")

        // The load-bearing pin: every replayed assistant tool call has a
        // matching tool message whose content is non-empty. An empty
        // content here is exactly issue #349's "no result or output back".
        let toolEntry = try XCTUnwrap(
            request.first { $0.role == "tool" && $0.toolCallId == "call_term" }
        )
        let content = try XCTUnwrap(toolEntry.content)
        XCTAssertFalse(content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertTrue(content.contains("tool_not_executed"), content)
    }

    // MARK: - 3. The Settings workspace folder reaches the hermes subprocess

    func testAgentWorkspaceFromSettingsReachesHermesLaunchEnvironment() throws {
        let workspace = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-349-workspace-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)

        let configuration = MTPLXAppConfiguration(hermesWorkspacePath: workspace.path)
        let integration = HermesIntegration(
            hermesHome: FileManager.default.temporaryDirectory
                .appendingPathComponent("mtplx-349-hermes-home-\(UUID().uuidString)", isDirectory: true)
        )

        XCTAssertEqual(
            HermesIntegration.resolvedWorkspacePath(configuration: configuration),
            workspace.path
        )
        let env = integration.launchEnvironment(configuration: configuration)
        XCTAssertEqual(env["HERMES_WORKSPACE"], workspace.path)
        XCTAssertEqual(env["TERMINAL_CWD"], workspace.path)
    }

    func testAgentWorkspaceFromSettingsReachesHermesTerminalCwdConfig() throws {
        let workspace = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-349-workspace-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        let home = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-349-hermes-home-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)

        let configuration = MTPLXAppConfiguration(hermesWorkspacePath: workspace.path)
        let integration = HermesIntegration(hermesHome: home)
        let result = try integration.sync(configuration: configuration)

        XCTAssertEqual(result.workspacePath, workspace.path)
        let configText = try String(
            contentsOf: URL(fileURLWithPath: result.configPath),
            encoding: .utf8
        )
        XCTAssertTrue(
            configText.contains("cwd: '\(workspace.path)'"),
            "terminal.cwd missing from hermes config:\n\(configText)"
        )
        let envText = try String(
            contentsOf: URL(fileURLWithPath: result.envPath),
            encoding: .utf8
        )
        XCTAssertTrue(envText.contains("TERMINAL_CWD="), envText)
        XCTAssertTrue(envText.contains(workspace.path), envText)
    }
}
