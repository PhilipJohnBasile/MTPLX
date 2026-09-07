import Foundation

// MARK: - ChatConversationTitle
//
// What a conversation is called before it has a name, and how it gets
// one. A new conversation is created with the ACTIVE language's
// "New Chat", so whether a title is still the placeholder cannot be
// decided by comparing it with any one display string: the guard used
// to compare with the English literal and never fired in eleven of the
// twelve shipped languages, leaving every chat untitled. Placeholder
// detection is therefore derived from the string tables themselves —
// every language's translation plus the English literal that seeded
// rows before localisation — so no stored flag (and no schema change)
// is needed, and a row created in one language is still recognised
// after the user switches to another.

public enum ChatConversationTitle {
    /// The table key every language's placeholder is translated from,
    /// and the literal older rows were created with.
    public static let placeholderKey = "New Chat"

    /// Words of the first message a derived title keeps.
    public static let derivedWordLimit = 5

    /// The placeholder in the active language.
    public static var placeholder: String {
        tr(placeholderKey)
    }

    /// Every string that means "no title yet", read from the shipped
    /// tables once. A language whose table is missing the key resolves
    /// to the key itself, which is already in the set.
    public static let placeholders: Set<String> = {
        var titles: Set<String> = [placeholderKey]
        for language in AppLanguage.allCases {
            titles.insert(L10n.string(placeholderKey, language: language))
        }
        return titles
    }()

    public static func isPlaceholder(_ title: String) -> Bool {
        placeholders.contains(title.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// The title a conversation takes from its first message: the first
    /// few words, or the placeholder when the text has no words.
    public static func derived(from text: String) -> String {
        let words = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: { $0.isWhitespace || $0.isNewline })
            .prefix(derivedWordLimit)
            .map { String($0) }
        let joined = words.joined(separator: " ")
        return joined.isEmpty ? placeholder : joined
    }
}

public extension ChatConversation {
    /// True while the conversation still carries a placeholder title in
    /// any shipped language (or the English literal), so it should take
    /// its name from the next message the user sends.
    var titleIsPlaceholder: Bool {
        ChatConversationTitle.isPlaceholder(title)
    }
}
