import Foundation

// MARK: - ChatTurnFailure
//
// Why an assistant turn ended in failure, persisted BESIDE the turn's
// `ChatTurnStats` inside the same `ChatMessage.statsJSON` blob. The
// blob is plain JSON and both decoders ignore keys they do not know,
// so the failure rides along as one additive key: the store schema is
// untouched and every turn persisted before this type existed decodes
// exactly as it did. The settled bubble reads it to label a failed
// reply with the daemon's own message instead of a bare
// "Interrupted reply".

public struct ChatTurnFailure: Codable, Hashable, Sendable {
    /// What the user reads: the daemon's `error.message` verbatim for a
    /// server-reported failure.
    public var errorMessage: String

    public init(errorMessage: String) {
        self.errorMessage = errorMessage
    }

    /// The failure a persisted `statsJSON` recorded, or nil for a turn
    /// that completed (or predates failure recording).
    public static func decode(fromStatsJSON json: String?) -> ChatTurnFailure? {
        guard let json,
            let data = json.data(using: .utf8),
            let failure = try? JSONDecoder().decode(ChatTurnFailure.self, from: data),
            !failure.errorMessage.isEmpty
        else { return nil }
        return failure
    }

    /// One `statsJSON` blob carrying `stats` and, when present, the
    /// failure. Merged at the JSON-object level so each type keeps
    /// decoding only its own keys. Nil when there is nothing to store.
    public static func statsJSON(stats: ChatTurnStats?, failure: ChatTurnFailure?) -> String? {
        var object: [String: Any] = [:]
        if let stats, let fields = jsonObject(encoding: stats) {
            object.merge(fields) { _, new in new }
        }
        if let failure, let fields = jsonObject(encoding: failure) {
            object.merge(fields) { _, new in new }
        }
        guard !object.isEmpty,
            let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return nil }
        return json
    }

    private static func jsonObject<T: Encodable>(encoding value: T) -> [String: Any]? {
        guard let data = try? JSONEncoder().encode(value),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return object
    }
}
