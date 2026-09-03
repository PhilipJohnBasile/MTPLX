import Foundation
import os

// MARK: - Public shapes
//
// Port of Aphanes V2's `NativeWebSearchService` — parallel DuckDuckGo
// HTML + Brave HTML scrape, URL-level dedupe, interleaved-rank merge.
// No API keys; no embedding reranker (MTPLX has no embedder, per
// product decision). DDG's main endpoint serves CAPTCHAs more often
// in 2026; the `lite.duckduckgo.com` fallback below is the resilient
// path and Brave runs in parallel as a silent second source.

public struct WebSearchRequest: Sendable, Hashable {
    public var query: String
    public var maxResults: Int

    public init(query: String, maxResults: Int = 5) {
        self.query = query
        self.maxResults = maxResults
    }
}

public struct WebSearchResult: Sendable, Hashable {
    public var title: String
    public var url: URL
    public var snippet: String

    public init(title: String, url: URL, snippet: String) {
        self.title = title
        self.url = url
        self.snippet = snippet
    }
}

/// Transport abstraction so tests can mock HTTP without touching the
/// network. Default implementation forwards to `URLSession.shared`.
public protocol WebTransport: Sendable {
    func data(for request: URLRequest) async throws -> (Data, URLResponse)
}

public struct URLSessionWebTransport: WebTransport {
    private let session: URLSession
    public init(session: URLSession = .shared) { self.session = session }
    public func data(for request: URLRequest) async throws -> (Data, URLResponse) {
        try await session.data(for: request)
    }
}

public enum WebSearchServiceError: LocalizedError {
    case invalidSearchURL
    case badServerResponse(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidSearchURL:
            return tr("Could not build a valid search URL.")
        case .badServerResponse(let statusCode):
            return tr("Search backend returned HTTP %@.", String(statusCode))
        }
    }
}

// MARK: - In-memory cache

/// Shared search-result cache used by `WebSearchService`. Single
/// process-wide actor; the same query within a session returns cached
/// results to keep the model from re-issuing duplicate searches.
public actor WebSearchCache {
    public static let shared = WebSearchCache()
    private var searchResults: [String: [WebSearchResult]] = [:]

    public init() {}

    public func cached(for key: String) -> [WebSearchResult]? {
        searchResults[key]
    }

    public func store(_ results: [WebSearchResult], for key: String) {
        searchResults[key] = results
    }

    public func clear() {
        searchResults.removeAll()
    }
}

// MARK: - WebSearchService

public final class WebSearchService: @unchecked Sendable {
    private let transport: any WebTransport
    private let cache: WebSearchCache
    private static let log = Logger(subsystem: "com.mtplx.app", category: "WebSearch")

    /// Recent macOS Safari UA. DDG / Brave both gate on this; a "curl/8"
    /// UA returns empty results or trips bot detection immediately.
    public static let browserUserAgent =
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"

    public init(
        transport: any WebTransport = URLSessionWebTransport(),
        cache: WebSearchCache = .shared
    ) {
        self.transport = transport
        self.cache = cache
    }

    public func search(_ request: WebSearchRequest) async throws -> [WebSearchResult] {
        let cacheKey = request.query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if let cached = await cache.cached(for: cacheKey) {
            return Array(cached.prefix(max(request.maxResults, 0)))
        }
        let limit = max(request.maxResults, 0)
        guard limit > 0 else { return [] }

        let providerLimit = max(limit, 10)

        async let ddgBatch = providerBatch(
            provider: .duckDuckGo,
            query: request.query
        ) {
            try await self.performDDGSearch(query: request.query, maxResults: providerLimit)
        }
        async let braveBatch = providerBatch(
            provider: .brave,
            query: request.query
        ) {
            try await self.performBraveSearch(query: request.query, maxResults: providerLimit)
        }
        let batches = await [ddgBatch, braveBatch]

        let merged = WebSearchService.mergeProviderBatches(batches, maxResults: providerLimit)
        if merged.isEmpty {
            Self.log.error("All search providers returned empty for '\(request.query.prefix(60), privacy: .public)'")
        }
        await cache.store(merged, for: cacheKey)
        return Array(merged.prefix(limit))
    }

    // MARK: - Provider call wrappers

    private func providerBatch(
        provider: SearchProvider,
        query: String,
        search: @escaping @Sendable () async throws -> [WebSearchResult]
    ) async -> ProviderResultBatch {
        do {
            let results = try await search()
            Self.log.info(
                "\(provider.displayName, privacy: .public) returned \(results.count, privacy: .public) results for '\(query.prefix(60), privacy: .public)'"
            )
            return ProviderResultBatch(
                provider: provider,
                results: results.enumerated().map { index, result in
                    MergedProviderCandidate(
                        provider: provider,
                        providerRank: index,
                        result: result
                    )
                }
            )
        } catch {
            Self.log.error(
                "\(provider.displayName, privacy: .public) failed for '\(query.prefix(60), privacy: .public)': \(error.localizedDescription, privacy: .public)"
            )
            return ProviderResultBatch(provider: provider, results: [])
        }
    }

    private func performDDGSearch(query: String, maxResults: Int) async throws -> [WebSearchResult] {
        let endpoints = [
            URL(string: "https://html.duckduckgo.com/html/")!,
            URL(string: "https://lite.duckduckgo.com/lite/")!,
        ]
        for endpoint in endpoints {
            var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)
            components?.queryItems = [URLQueryItem(name: "q", value: query)]
            guard let url = components?.url else { continue }

            var urlRequest = URLRequest(url: url)
            urlRequest.setValue(Self.browserUserAgent, forHTTPHeaderField: "User-Agent")
            urlRequest.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")
            urlRequest.setValue("en-US,en;q=0.9", forHTTPHeaderField: "Accept-Language")
            urlRequest.setValue("https://duckduckgo.com/", forHTTPHeaderField: "Referer")

            let (data, response) = try await transport.data(for: urlRequest)
            try Self.validate(response: response)

            let html = String(decoding: data, as: UTF8.self)
            if Self.isDDGBotDetection(html) {
                Self.log.info("DDG bot detection triggered on \(endpoint.host ?? "")")
                continue
            }
            let results = Self.parseDDGResults(from: html, limit: maxResults)
            if !results.isEmpty { return results }
        }
        return []
    }

    private func performBraveSearch(query: String, maxResults: Int) async throws -> [WebSearchResult] {
        var components = URLComponents(string: "https://search.brave.com/search")
        components?.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "source", value: "web"),
        ]
        guard let url = components?.url else { return [] }

        var urlRequest = URLRequest(url: url)
        urlRequest.setValue(Self.browserUserAgent, forHTTPHeaderField: "User-Agent")
        urlRequest.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")
        urlRequest.setValue("en-US,en;q=0.9", forHTTPHeaderField: "Accept-Language")

        let (data, response) = try await transport.data(for: urlRequest)
        try Self.validate(response: response)
        let html = String(decoding: data, as: UTF8.self)
        return Self.parseBraveResults(from: html, limit: maxResults)
    }
}

// MARK: - Internal: provider taxonomy and merging

extension WebSearchService {
    enum SearchProvider: String, Sendable {
        case duckDuckGo
        case brave

        var displayName: String {
            switch self {
            case .duckDuckGo: return "DuckDuckGo"
            case .brave: return "Brave"
            }
        }
        var sortPriority: Int {
            switch self {
            case .duckDuckGo: return 0
            case .brave: return 1
            }
        }
    }

    struct MergedProviderCandidate: Sendable {
        let provider: SearchProvider
        let providerRank: Int
        let result: WebSearchResult
    }

    struct ProviderResultBatch: Sendable {
        let provider: SearchProvider
        let results: [MergedProviderCandidate]
    }

    static func preferredCandidate(
        current: MergedProviderCandidate,
        replacement: MergedProviderCandidate
    ) -> MergedProviderCandidate {
        if current.providerRank != replacement.providerRank {
            return current.providerRank < replacement.providerRank ? current : replacement
        }
        let currentSnippetLength = current.result.snippet.count
        let replacementSnippetLength = replacement.result.snippet.count
        if currentSnippetLength != replacementSnippetLength {
            return currentSnippetLength > replacementSnippetLength ? current : replacement
        }
        return current.provider.sortPriority <= replacement.provider.sortPriority ? current : replacement
    }

    static func mergeProviderBatches(
        _ batches: [ProviderResultBatch],
        maxResults: Int
    ) -> [WebSearchResult] {
        guard maxResults > 0 else { return [] }

        var dedupedByURL: [String: MergedProviderCandidate] = [:]
        for batch in batches {
            for candidate in batch.results {
                let key = candidate.result.url.absoluteString
                if let existing = dedupedByURL[key] {
                    dedupedByURL[key] = preferredCandidate(current: existing, replacement: candidate)
                } else {
                    dedupedByURL[key] = candidate
                }
            }
        }

        let groupedByRank = Dictionary(grouping: dedupedByURL.values) { $0.providerRank }
        let orderedRanks = groupedByRank.keys.sorted()
        var merged: [WebSearchResult] = []
        merged.reserveCapacity(maxResults)

        for rank in orderedRanks {
            let group = (groupedByRank[rank] ?? []).sorted { lhs, rhs in
                let lhsSnippetLength = lhs.result.snippet.count
                let rhsSnippetLength = rhs.result.snippet.count
                if lhsSnippetLength != rhsSnippetLength {
                    return lhsSnippetLength > rhsSnippetLength
                }
                if lhs.provider.sortPriority != rhs.provider.sortPriority {
                    return lhs.provider.sortPriority < rhs.provider.sortPriority
                }
                return lhs.result.url.absoluteString < rhs.result.url.absoluteString
            }
            for candidate in group {
                merged.append(candidate.result)
                if merged.count == maxResults { return merged }
            }
        }
        return merged
    }

    static func validate(response: URLResponse) throws {
        if let httpResponse = response as? HTTPURLResponse,
            !(200...299).contains(httpResponse.statusCode)
        {
            throw WebSearchServiceError.badServerResponse(httpResponse.statusCode)
        }
    }

    static func isDDGBotDetection(_ html: String) -> Bool {
        html.contains("cc=botnet") || html.contains("anomaly-modal") || html.contains("bots use DuckDuckGo")
    }
}

// MARK: - HTML parsing (verbatim from Aphanes)

extension WebSearchService {
    static func parseDDGResults(from html: String, limit: Int) -> [WebSearchResult] {
        guard limit > 0 else { return [] }
        let anchorPattern = #"<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#
        let snippetPattern = #"<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>"#

        guard
            let anchorRegex = try? NSRegularExpression(pattern: anchorPattern, options: [.dotMatchesLineSeparators]),
            let snippetRegex = try? NSRegularExpression(pattern: snippetPattern, options: [.dotMatchesLineSeparators])
        else { return [] }

        let fullRange = NSRange(html.startIndex..., in: html)
        let anchorMatches = anchorRegex.matches(in: html, options: [], range: fullRange)
        guard !anchorMatches.isEmpty else { return [] }

        return anchorMatches.prefix(limit).compactMap { match in
            guard
                let hrefRange = Range(match.range(at: 1), in: html),
                let titleRange = Range(match.range(at: 2), in: html)
            else { return nil }

            let href = String(html[hrefRange])
            guard let resolvedURL = resolveDDGResultURL(from: href) else { return nil }

            let title = cleanHTMLFragment(String(html[titleRange]))
            let snippetRangeStart = match.range.location + match.range.length
            let snippetSearchRange = NSRange(
                location: snippetRangeStart,
                length: max(fullRange.length - snippetRangeStart, 0)
            )
            let snippetMatch = snippetRegex.firstMatch(in: html, options: [], range: snippetSearchRange)
            let snippet = snippetMatch.flatMap { m in
                Range(m.range(at: 1), in: html).map { cleanHTMLFragment(String(html[$0])) }
            } ?? ""
            return WebSearchResult(title: title, url: resolvedURL, snippet: snippet)
        }
    }

    static func parseBraveResults(from html: String, limit: Int) -> [WebSearchResult] {
        guard limit > 0 else { return [] }

        let snippetBlockPattern = #"<div[^>]*class="[^"]*snippet\s[^"]*"[^>]*data-pos="(\d+)"[^>]*>(.*?)</div>\s*</div>\s*</div>"#
        guard let blockRegex = try? NSRegularExpression(pattern: snippetBlockPattern, options: [.dotMatchesLineSeparators]) else {
            return parseBraveFallback(from: html, limit: limit)
        }

        let fullRange = NSRange(html.startIndex..., in: html)
        let blocks = blockRegex.matches(in: html, options: [], range: fullRange)
        if blocks.isEmpty { return parseBraveFallback(from: html, limit: limit) }

        let titlePattern = #"<div[^>]*class="[^"]*search-snippet-title[^"]*"[^>]*>(.*?)</div>"#
        let hrefPattern = #"<a[^>]*href="(https?://[^"]+)"[^>]*class="[^"]*svelte-[^"]*"[^>]*>"#
        let contentPattern = #"<div[^>]*class="content[^"]*"[^>]*>(.*?)</div>"#

        let titleRegex = try? NSRegularExpression(pattern: titlePattern, options: [.dotMatchesLineSeparators])
        let hrefRegex = try? NSRegularExpression(pattern: hrefPattern, options: [.dotMatchesLineSeparators])
        let contentRegex = try? NSRegularExpression(pattern: contentPattern, options: [.dotMatchesLineSeparators])

        var results: [WebSearchResult] = []
        for block in blocks.prefix(limit) {
            guard let blockRange = Range(block.range(at: 2), in: html) else { continue }
            let blockHTML = String(html[blockRange])
            let blockNSRange = NSRange(blockHTML.startIndex..., in: blockHTML)

            let title: String
            if let m = titleRegex?.firstMatch(in: blockHTML, options: [], range: blockNSRange),
                let r = Range(m.range(at: 1), in: blockHTML) {
                title = cleanHTMLFragment(String(blockHTML[r]))
            } else { continue }

            let urlString: String
            if let m = hrefRegex?.firstMatch(in: blockHTML, options: [], range: blockNSRange),
                let r = Range(m.range(at: 1), in: blockHTML) {
                urlString = String(blockHTML[r])
            } else { continue }
            guard let url = URL(string: urlString) else { continue }

            let snippet: String
            if let m = contentRegex?.firstMatch(in: blockHTML, options: [], range: blockNSRange),
                let r = Range(m.range(at: 1), in: blockHTML) {
                snippet = cleanHTMLFragment(String(blockHTML[r]))
            } else { snippet = "" }

            results.append(WebSearchResult(title: title, url: url, snippet: snippet))
        }
        return results
    }

    static func parseBraveFallback(from html: String, limit: Int) -> [WebSearchResult] {
        let titlePattern = #"<div[^>]*class="[^"]*search-snippet-title[^"]*"[^>]*title="([^"]*)"[^>]*>"#
        let hrefPattern = #"<a[^>]*href="(https?://[^"]+)"[^>]*>[^<]*<div[^>]*class="[^"]*site-name"#
        guard
            let titleRegex = try? NSRegularExpression(pattern: titlePattern, options: [.dotMatchesLineSeparators]),
            let hrefRegex = try? NSRegularExpression(pattern: hrefPattern, options: [.dotMatchesLineSeparators])
        else { return [] }

        let fullRange = NSRange(html.startIndex..., in: html)
        let titleMatches = titleRegex.matches(in: html, options: [], range: fullRange)
        let hrefMatches = hrefRegex.matches(in: html, options: [], range: fullRange)
        let count = min(min(titleMatches.count, hrefMatches.count), limit)

        var results: [WebSearchResult] = []
        for i in 0..<count {
            guard let titleRange = Range(titleMatches[i].range(at: 1), in: html),
                let hrefRange = Range(hrefMatches[i].range(at: 1), in: html),
                let url = URL(string: String(html[hrefRange]))
            else { continue }
            let title = decodeHTMLEntities(in: String(html[titleRange]))
            results.append(WebSearchResult(title: title, url: url, snippet: ""))
        }
        return results
    }

    static func resolveDDGResultURL(from href: String) -> URL? {
        if let components = URLComponents(string: href),
            let uddg = components.queryItems?.first(where: { $0.name == "uddg" })?.value,
            let resolved = URL(string: uddg.removingPercentEncoding ?? uddg) {
            return resolved
        }
        return URL(string: decodeHTMLEntities(in: href))
    }
}

// MARK: - Shared HTML helpers (used by both search parsing and URL fetch)

enum WebHTML {
    static func cleanHTMLFragment(_ fragment: String) -> String {
        let withoutTags = fragment.replacingOccurrences(of: #"<[^>]+>"#, with: " ", options: [.regularExpression])
        return normalizeWhitespace(decodeHTMLEntities(in: withoutTags))
    }

    static func decodeHTMLEntities(in value: String) -> String {
        value
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&apos;", with: "'")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&nbsp;", with: " ")
    }

    static func normalizeWhitespace(_ value: String) -> String {
        value
            .replacingOccurrences(of: #"\s+"#, with: " ", options: [.regularExpression])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func extractHTMLTitle(from html: String) -> String? {
        let pattern = #"<title[^>]*>(.*?)</title>"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators]) else {
            return nil
        }
        let range = NSRange(html.startIndex..., in: html)
        guard
            let match = regex.firstMatch(in: html, options: [], range: range),
            let titleRange = Range(match.range(at: 1), in: html)
        else { return nil }
        return cleanHTMLFragment(String(html[titleRange]))
    }

    static func extractReadableContent(from html: String) -> String {
        var stripped = html
        let removals = [
            #"<script\b[^>]*>.*?</script>"#,
            #"<style\b[^>]*>.*?</style>"#,
            #"<noscript\b[^>]*>.*?</noscript>"#,
        ]
        for pattern in removals {
            stripped = stripped.replacingOccurrences(
                of: pattern, with: " ",
                options: [.regularExpression, .caseInsensitive]
            )
        }
        let structuralTags = [
            #"</p>"#, #"<br\s*/?>"#, #"</div>"#, #"</li>"#, #"</h[1-6]>"#,
        ]
        for tag in structuralTags {
            stripped = stripped.replacingOccurrences(of: tag, with: "\n", options: [.regularExpression, .caseInsensitive])
        }
        stripped = stripped.replacingOccurrences(of: #"<[^>]+>"#, with: " ", options: [.regularExpression])
        stripped = decodeHTMLEntities(in: stripped)

        let lines = stripped
            .components(separatedBy: .newlines)
            .map(normalizeWhitespace)
            .filter { !$0.isEmpty }
        return lines.joined(separator: "\n")
    }
}

extension WebSearchService {
    static func cleanHTMLFragment(_ fragment: String) -> String {
        WebHTML.cleanHTMLFragment(fragment)
    }
    static func decodeHTMLEntities(in value: String) -> String {
        WebHTML.decodeHTMLEntities(in: value)
    }
}
