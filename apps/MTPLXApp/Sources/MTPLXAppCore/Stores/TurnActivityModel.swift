import Foundation

// MARK: - TurnActivityModel
//
// Pure presentation model for the assistant turn's activity strip —
// the row of equal-width chips ("Thought", "Searched the web") that
// sits above the answer, plus which detail well should be open. One
// model serves BOTH the live streaming surface and the settled
// transcript bubble, so the moment a turn finishes the strip doesn't
// change shape — only its captions settle (durations, ×N counts).
//
// Kept in AppCore (no SwiftUI) so chip composition and the
// phase→well auto-follow rules are unit-testable.

public struct TurnActivityModel: Equatable {

    /// Which detail well is open beneath the chip row. `.none` renders
    /// the strip as bare chips (the settled default, and the streaming
    /// state once answer tokens start).
    public enum Detail: Equatable {
        case none
        case thought
        case search
    }

    public struct Chip: Equatable, Identifiable {
        public enum Kind: String, Equatable {
            case thought
            case search

            public var detail: Detail {
                switch self {
                case .thought: return .thought
                case .search: return .search
                }
            }
        }

        public let kind: Kind
        public let systemName: String
        public let label: String
        /// Dim trailing annotation: "12.4s" on a settled thought chip,
        /// "×3" on a settled search chip. Nil while live.
        public let caption: String?
        /// Live chips pulse (indicator dots) and read brighter.
        public let isLive: Bool

        public var id: String { kind.rawValue }

        public init(
            kind: Kind,
            systemName: String,
            label: String,
            caption: String? = nil,
            isLive: Bool = false
        ) {
            self.kind = kind
            self.systemName = systemName
            self.label = label
            self.caption = caption
            self.isLive = isLive
        }
    }

    public let chips: [Chip]

    public var isEmpty: Bool { chips.isEmpty }

    public func hasChip(_ kind: Chip.Kind) -> Bool {
        chips.contains { $0.kind == kind }
    }

    public init(chips: [Chip]) {
        self.chips = chips
    }

    // MARK: - Live (in-flight turn)

    /// Chips for the streaming surface. The thought chip exists from
    /// the moment the request is in flight (so nothing pops in later);
    /// the search chip appears beside it when the first tool call of
    /// the turn dispatches and stays for the rest of the turn.
    public static func live(
        phase: StreamingPhase,
        hasReasoning: Bool,
        traces: [PendingToolTrace]
    ) -> TurnActivityModel {
        var chips: [Chip] = []

        let thoughtIsLive = phase == .thinking || phase == .generating
        if hasReasoning || thoughtIsLive {
            let label: String
            if thoughtIsLive {
                label = (phase == .generating && !hasReasoning) ? tr("Generating") : "Thinking"
            } else {
                label = tr("Thought")
            }
            chips.append(
                Chip(kind: .thought, systemName: "brain", label: label, isLive: thoughtIsLive)
            )
        }

        if !traces.isEmpty {
            let searchIsLive = phase == .searching || phase == .reading
                || traces.contains { $0.status == .pending }
            let label: String
            var caption: String?
            if searchIsLive {
                label = phase == .reading ? tr("Reading") : "Searching"
            } else {
                // Only calls that actually ran count as searches or
                // pages read; failures are counted on their own so the
                // chip never reports a failed search as "Searched".
                (label, caption) = Self.settledSearchLabel(
                    searchCount: traces.filter { $0.name == "web_search" && $0.status == .success }.count,
                    fetchedPageCount: traces.filter { $0.name == "fetch_url" && $0.status == .success }.count,
                    hasOtherToolActivity: true,
                    failedSearchCount: traces.filter { $0.name == "web_search" && $0.status == .failed }.count,
                    failedFetchCount: traces.filter { $0.name == "fetch_url" && $0.status == .failed }.count
                )
            }
            chips.append(
                Chip(
                    kind: .search,
                    systemName: searchIsLive && phase == .reading ? "doc.text" : "globe",
                    label: label,
                    caption: caption,
                    isLive: searchIsLive
                )
            )
        }

        return TurnActivityModel(chips: chips)
    }

    // MARK: - Settled (persisted turn)

    /// `searchCount` / `fetchedPageCount` are calls that ran; failed
    /// calls arrive separately so they are labelled as failures rather
    /// than counted as work done.
    public static func settled(
        hasThought: Bool,
        thinkingTimeMs: Int?,
        searchCount: Int,
        fetchedPageCount: Int,
        hasOtherToolActivity: Bool,
        failedSearchCount: Int = 0,
        failedFetchCount: Int = 0
    ) -> TurnActivityModel {
        var chips: [Chip] = []
        if hasThought {
            chips.append(
                Chip(
                    kind: .thought,
                    systemName: "brain",
                    label: tr("Thought"),
                    caption: thinkingTimeMs.map(Self.formatDuration)
                )
            )
        }
        if searchCount > 0 || fetchedPageCount > 0 || hasOtherToolActivity
            || failedSearchCount > 0 || failedFetchCount > 0
        {
            let (label, caption) = Self.settledSearchLabel(
                searchCount: searchCount,
                fetchedPageCount: fetchedPageCount,
                hasOtherToolActivity: hasOtherToolActivity,
                failedSearchCount: failedSearchCount,
                failedFetchCount: failedFetchCount
            )
            chips.append(
                Chip(kind: .search, systemName: "globe", label: label, caption: caption)
            )
        }
        return TurnActivityModel(chips: chips)
    }

    private static func settledSearchLabel(
        searchCount: Int,
        fetchedPageCount: Int,
        hasOtherToolActivity: Bool,
        failedSearchCount: Int,
        failedFetchCount: Int
    ) -> (label: String, caption: String?) {
        let failedCount = failedSearchCount + failedFetchCount
        if searchCount > 0 {
            return ("Searched", Self.countCaption(searchCount, failed: failedCount))
        }
        if fetchedPageCount > 0 {
            return (
                fetchedPageCount == 1 ? tr("Read a page") : "Read pages",
                Self.countCaption(fetchedPageCount, failed: failedCount)
            )
        }
        // Nothing ran: the chip names the failure instead of "Searched".
        if failedSearchCount > 0 {
            return (tr("Search failed"), Self.countCaption(failedSearchCount, failed: failedFetchCount))
        }
        if failedFetchCount > 0 {
            return (tr("Fetch failed"), Self.countCaption(failedFetchCount, failed: 0))
        }
        return ("Used tools", nil)
    }

    /// "×3", "×3 · 1 failed", "1 failed", or nil when there is nothing
    /// beyond the label to say.
    private static func countCaption(_ count: Int, failed: Int) -> String? {
        var parts: [String] = []
        if count > 1 { parts.append("×\(count)") }
        if failed > 0 { parts.append(tr("%lld failed", failed)) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    // MARK: - Auto-follow
    //
    // While streaming, the open well tracks what the model is doing:
    // thinking expands the thought well (collapsing search), a tool
    // round expands the search well (collapsing thought), and the
    // first answer token closes both so the reply gets the stage.

    public static func autoDetail(for phase: StreamingPhase) -> Detail {
        switch phase {
        case .thinking, .generating:
            return .thought
        case .searching, .reading:
            return .search
        case .answering, .finalizing, .idle:
            return .none
        }
    }

    public static func formatDuration(_ ms: Int) -> String {
        let seconds = Double(ms) / 1000.0
        if seconds < 1.0 { return "\(ms) ms" }
        return String(format: "%.1fs", seconds)
    }
}
