import Foundation
import os

// MARK: - MTPLXChatToolFactory
//
// Builds the OpenAI-style tool surface the in-app chat exposes to the
// model and routes each tool call to the right service. The schemas
// (`web_search`, `fetch_url`) match Aphanes V2 verbatim so the model's
// learned tool-use behaviour transfers directly.
//
// Differences from Aphanes (deliberate, per plan):
//   - No memory/RAG tools. MTPLX has no embedder; those tools depend
//     on one.
//   - No comparison-query expansion or embedding rerank. Results are
//     DDG+Brave merge-order only.
//   - Tool-round policy is enforced by `ChatViewModel`, not here; the
//     factory dispatches one call at a time and is stateless except for
//     the Jaccard duplicate-query guard.

public struct MTPLXChatToolFactory: Sendable {
    public let webSearch: WebSearchService
    public let urlFetcher: URLFetcher
    public let session: ToolSessionState

    private static let log = Logger(subsystem: "com.mtplx.app", category: "ChatTools")

    public init(
        webSearch: WebSearchService = WebSearchService(),
        urlFetcher: URLFetcher = URLFetcher(),
        session: ToolSessionState = ToolSessionState()
    ) {
        self.webSearch = webSearch
        self.urlFetcher = urlFetcher
        self.session = session
    }

    // MARK: - Tool definitions

    /// OpenAI-shape tool definitions the chat client puts on the wire.
    /// Returns typed `ChatRequestTool` values so the viewmodel can drop
    /// them straight into `ChatRequest.tools`.
    public func toolDefinitions(
        webSearchEnabled: Bool = true,
        workspaceRoot: String? = nil
    ) -> [ChatRequestTool] {
        var definitions: [ChatRequestTool] = []
        if webSearchEnabled {
            definitions += [webSearchToolDefinition(), fetchURLToolDefinition()]
        }
        if workspaceRoot != nil {
            definitions += MTPLXWorkspaceToolService().toolDefinitions()
        }
        return definitions
    }

    public func requiresApproval(for name: String) -> Bool {
        requiresApproval(for: name, policy: [:])
    }

    public func requiresApproval(for name: String, policy: [String: String]) -> Bool {
        guard let key = Self.workspacePolicyKey(for: name) else { return false }
        return policy[key]?.lowercased() != "allow"
    }

    public static func workspacePolicyKey(for name: String) -> String? {
        switch name {
        case "list_files", "read_file", "inspect_repo", "git_status", "git_diff":
            return "read"
        case "search_files":
            return "search"
        case "write_file", "apply_patch":
            return "write"
        case "run_tests", "run_command":
            return "terminal"
        case "web_search", "fetch_url":
            return "browser"
        default:
            return nil
        }
    }

    /// Wakes the per-turn state machine — call once at the start of
    /// each user turn so the Jaccard guard sees a fresh slate.
    public func beginTurn() async {
        await session.reset()
    }

    // MARK: - Dispatch

    /// Routes a tool call. Returns the raw JSON string to send back as
    /// the assistant's tool-result message content.
    public func dispatch(name: String, argumentsJSON: String) async -> String {
        switch name {
        case "web_search":
            return await dispatchWebSearch(argumentsJSON: argumentsJSON)
        case "fetch_url":
            return await dispatchFetchURL(argumentsJSON: argumentsJSON)
        default:
            Self.log.warning("Unknown tool name: \(name, privacy: .public)")
            return jsonObject([
                "error": "unknown_tool",
                "tool": name,
                "note": "Tool not implemented in MTPLX chat; answer from knowledge.",
            ])
        }
    }

    // MARK: - web_search

    private func dispatchWebSearch(argumentsJSON: String) async -> String {
        let query = parseQuery(from: argumentsJSON).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else {
            return jsonObject([
                "error": "empty_query",
                "note": "web_search called with empty query; answer from knowledge.",
            ])
        }

        switch await session.begin(query: query) {
        case .proceed:
            break
        case .duplicate(let previous, let warningCount):
            Self.log.info(
                "Skipping duplicate web_search query='\(query, privacy: .public)' previous='\(previous, privacy: .public)' warnings=\(warningCount, privacy: .public)"
            )
            return jsonObject([
                "query": query,
                "previous_query": previous,
                "warning_count": warningCount,
                "note": "Query is too similar to a previous search this turn. Use the earlier results instead of repeating the search.",
            ])
        case .disabled:
            Self.log.info("web_search disabled for the remainder of this turn")
            return jsonObject([
                "query": query,
                "note": "web_search is disabled for the rest of this turn. Answer from knowledge or the previously fetched sources.",
            ])
        }

        let searchRequest = WebSearchRequest(query: query, maxResults: 5)
        let searchResults: [WebSearchResult]
        do {
            searchResults = try await webSearch.search(searchRequest)
        } catch {
            Self.log.error("web_search failed: \(error.localizedDescription, privacy: .public)")
            return jsonObject([
                "query": query,
                "error": "search_failed",
                "detail": error.localizedDescription,
                "note": "Search backend errored; answer from knowledge and do not retry.",
            ])
        }

        guard !searchResults.isEmpty else {
            return jsonObject([
                "query": query,
                "results": [] as [Any],
                "note": "No results. Answer the user's question from your knowledge.",
            ])
        }

        // Fetch full readable text for the top 3 URLs in parallel.
        // Limit is conservative on purpose — beyond 3 we'd inflate the
        // tool-result body past the model's useful attention budget.
        let fetchCount = min(3, searchResults.count)
        let fetched = await withTaskGroup(of: (URL, URLFetchResult?).self) { group in
            for result in searchResults.prefix(fetchCount) {
                group.addTask { @Sendable [urlFetcher] in
                    do {
                        let page = try await urlFetcher.fetch(URLFetchRequest(url: result.url))
                        return (result.url, page)
                    } catch {
                        return (result.url, nil)
                    }
                }
            }
            var results: [URL: URLFetchResult?] = [:]
            for await pair in group {
                results[pair.0] = pair.1
            }
            return results
        }

        let resultObjects: [[String: Any]] = searchResults.map { result in
            var dict: [String: Any] = [
                "title": result.title,
                "url": result.url.absoluteString,
                "snippet": result.snippet,
                "host": result.url.host ?? "",
            ]
            if let page = fetched[result.url] ?? nil {
                dict["page_content"] = page.content
            }
            return dict
        }

        return jsonObject([
            "query": query,
            "results": resultObjects,
        ])
    }

    // MARK: - fetch_url

    private func dispatchFetchURL(argumentsJSON: String) async -> String {
        let rawURL = parseString(from: argumentsJSON, key: "url")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: rawURL), let scheme = url.scheme,
            scheme == "http" || scheme == "https"
        else {
            return jsonObject([
                "error": "invalid_url",
                "url": rawURL,
                "note": "fetch_url requires an http(s) URL.",
            ])
        }

        do {
            let result = try await urlFetcher.fetch(URLFetchRequest(url: url))
            return jsonObject([
                "url": result.url.absoluteString,
                "title": result.title ?? "",
                "content": result.content,
            ])
        } catch {
            Self.log.error("fetch_url failed for \(url, privacy: .public): \(error.localizedDescription, privacy: .public)")
            return jsonObject([
                "error": "fetch_failed",
                "url": url.absoluteString,
                "detail": error.localizedDescription,
                "note": "Could not fetch the URL; do not retry the same URL this turn.",
            ])
        }
    }

    // MARK: - Schema definitions

    private func webSearchToolDefinition() -> ChatRequestTool {
        ChatRequestTool(
            function: ChatRequestToolDefinition(
                name: "web_search",
                description: "Search the web and automatically read the strongest current sources. Use this for current facts, product comparisons, recent releases, reputational questions, or any claim you need to verify.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object([
                            "type": .string("string"),
                            "description": .string("The search query"),
                        ]),
                    ]),
                    "required": .array([.string("query")]),
                ])
            )
        )
    }

    private func fetchURLToolDefinition() -> ChatRequestTool {
        ChatRequestTool(
            function: ChatRequestToolDefinition(
                name: "fetch_url",
                description: "Fetch and extract readable text content from a URL. Use this when the user provides a URL and wants to know what is on that page.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "url": .object([
                            "type": .string("string"),
                            "description": .string("The URL to fetch"),
                        ]),
                    ]),
                    "required": .array([.string("url")]),
                ])
            )
        )
    }

    // MARK: - JSON parsing helpers

    private func parseQuery(from argumentsJSON: String) -> String {
        parseString(from: argumentsJSON, key: "query")
    }

    private func parseString(from argumentsJSON: String, key: String) -> String {
        guard let data = argumentsJSON.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return "" }
        return (object[key] as? String) ?? ""
    }

    private func jsonObject(_ dict: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: dict, options: []),
            let text = String(data: data, encoding: .utf8)
        else {
            return "{\"error\":\"json_encode_failed\"}"
        }
        return text
    }
}

// MARK: - Per-turn duplicate-query guard
//
// Mirrors Aphanes' `SearchSessionState`. After two near-duplicate
// `web_search` queries within one turn, the third call is denied with
// an explicit JSON note so the model can recover instead of spinning.

public actor ToolSessionState {
    /// Lower-case, whitespace-stripped query strings already issued
    /// this turn. Used by the Jaccard guard.
    private var seenQueries: [String] = []
    private var warningCount: Int = 0
    private var disabled: Bool = false
    private static let jaccardThreshold: Double = 0.85
    private static let maxWarnings: Int = 2

    public init() {}

    public func reset() {
        seenQueries.removeAll()
        warningCount = 0
        disabled = false
    }

    public enum Decision: Sendable {
        case proceed
        case duplicate(previous: String, warningCount: Int)
        case disabled
    }

    public func begin(query rawQuery: String) -> Decision {
        if disabled { return .disabled }
        let normalized = rawQuery.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return .proceed }
        for previous in seenQueries {
            if jaccard(previous, normalized) >= Self.jaccardThreshold {
                warningCount += 1
                if warningCount >= Self.maxWarnings {
                    disabled = true
                }
                return .duplicate(previous: previous, warningCount: warningCount)
            }
        }
        seenQueries.append(normalized)
        return .proceed
    }

    private func jaccard(_ lhs: String, _ rhs: String) -> Double {
        let left = Set(lhs.split(separator: " ").map { String($0) })
        let right = Set(rhs.split(separator: " ").map { String($0) })
        let intersection = left.intersection(right).count
        let union = left.union(right).count
        guard union > 0 else { return 0 }
        return Double(intersection) / Double(union)
    }
}
