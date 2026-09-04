import Foundation

// MARK: - AppLanguage
//
// The languages the app ships UI strings for. Declaration order is the
// order the language picker shows them in. `rawValue` is the BCP 47
// code and doubles as the `.lproj` directory name in the bundle, so a
// new language is one enum case plus one `Localizable.strings` table.
//
// English is the base: every string key in the code IS the English
// text, so a missing translation degrades to English, never to a raw
// identifier.

public enum AppLanguage: String, CaseIterable, Codable, Hashable, Identifiable, Sendable {
    case english = "en"
    case simplifiedChinese = "zh-Hans"
    case spanish = "es"
    case hindi = "hi"
    case arabic = "ar"
    case brazilianPortuguese = "pt-BR"
    case french = "fr"
    case russian = "ru"
    case japanese = "ja"
    case german = "de"
    case korean = "ko"
    case indonesian = "id"

    public static let base: AppLanguage = .english

    public var id: String { rawValue }

    /// BCP 47 code, also the `.lproj` directory name.
    public var code: String { rawValue }

    /// The language's own name for itself, shown as the row title.
    public var nativeName: String {
        switch self {
        case .english: return "English"
        case .simplifiedChinese: return "简体中文"
        case .spanish: return "Español"
        case .hindi: return "हिन्दी"
        case .arabic: return "العربية"
        case .brazilianPortuguese: return "Português (Brasil)"
        case .french: return "Français"
        case .russian: return "Русский"
        case .japanese: return "日本語"
        case .german: return "Deutsch"
        case .korean: return "한국어"
        case .indonesian: return "Bahasa Indonesia"
        }
    }

    /// English name, shown as the row caption so any reader can find a
    /// language whose script they cannot read.
    public var englishName: String {
        switch self {
        case .english: return "English"
        case .simplifiedChinese: return "Chinese (Simplified)"
        case .spanish: return "Spanish"
        case .hindi: return "Hindi"
        case .arabic: return "Arabic"
        case .brazilianPortuguese: return "Portuguese (Brazil)"
        case .french: return "French"
        case .russian: return "Russian"
        case .japanese: return "Japanese"
        case .german: return "German"
        case .korean: return "Korean"
        case .indonesian: return "Indonesian"
        }
    }

    /// Regional flag emoji for the picker rows.
    public var flag: String {
        switch self {
        case .english: return "🇺🇸"
        case .simplifiedChinese: return "🇨🇳"
        case .spanish: return "🇪🇸"
        case .hindi: return "🇮🇳"
        case .arabic: return "🇸🇦"
        case .brazilianPortuguese: return "🇧🇷"
        case .french: return "🇫🇷"
        case .russian: return "🇷🇺"
        case .japanese: return "🇯🇵"
        case .german: return "🇩🇪"
        case .korean: return "🇰🇷"
        case .indonesian: return "🇮🇩"
        }
    }

    /// Arabic lays the whole UI out right-to-left; everything else is
    /// left-to-right. The host maps this onto SwiftUI's `LayoutDirection`.
    public var isRightToLeft: Bool {
        self == .arabic
    }

    /// Locale for number, date and unit formatting in this language.
    public var locale: Locale {
        Locale(identifier: rawValue)
    }

    // MARK: Lookup

    /// Exact (case-insensitive) code match: "zh-hans" resolves, "zh" does
    /// not. Use `bestMatch(preferredLanguages:)` for fuzzy resolution.
    public init?(code: String) {
        let needle = code.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let match = AppLanguage.allCases.first(where: { $0.rawValue.lowercased() == needle }) else {
            return nil
        }
        self = match
    }

    /// Picks the app language for a user's macOS preferred-language list
    /// (`Locale.preferredLanguages` order). Matching follows Apple's
    /// bundle-localization rules: language code plus script, region
    /// ignored. So "pt-PT" lands on Brazilian Portuguese (the only
    /// Portuguese shipped), "es-419" on Spanish, "zh-CN" on Simplified
    /// Chinese, while "zh-Hant-TW" is a different script and falls
    /// through to the next preferred language. English when nothing
    /// matches.
    public static func bestMatch(preferredLanguages: [String]) -> AppLanguage {
        for identifier in preferredLanguages {
            if let exact = AppLanguage(code: identifier) {
                return exact
            }
            let wanted = languageAndScript(of: identifier)
            if let match = AppLanguage.allCases.first(where: { languageAndScript(of: $0.rawValue) == wanted }) {
                return match
            }
        }
        return .base
    }

    /// "zh-CN" -> ("zh", "Hans"), "pt-PT" -> ("pt", "Latn"), "sr" -> ("sr", "Cyrl").
    /// ICU's likely-subtags data fills the script in, so bare codes and
    /// region-only codes compare against the same key as scripted ones.
    private static func languageAndScript(of identifier: String) -> String {
        let language = Locale.Language(identifier: identifier)
        let maximal = Locale.Language(identifier: language.maximalIdentifier)
        let code = maximal.languageCode?.identifier.lowercased() ?? identifier.lowercased()
        let script = maximal.script?.identifier.lowercased() ?? ""
        return "\(code)-\(script)"
    }

    // MARK: Picker search

    /// Rows whose native name, English name or code contain `query`,
    /// ignoring case, diacritics and character width. An empty or
    /// whitespace-only query keeps every language in declaration order.
    public static func matching(_ query: String, in languages: [AppLanguage] = AppLanguage.allCases) -> [AppLanguage] {
        let needle = searchFolded(query)
        guard !needle.isEmpty else { return languages }
        return languages.filter { $0.matches(foldedQuery: needle) }
    }

    /// Whether this language matches an already-folded search query.
    /// Fold the user's text once with `searchFolded(_:)` and reuse it.
    /// Both sides are folded first and then compared literally: a
    /// collation-aware search refuses to end a match inside an Indic
    /// conjunct cluster, so "हिन" would never find "हिन्दी".
    public func matches(foldedQuery needle: String) -> Bool {
        guard !needle.isEmpty else { return true }
        return [nativeName, englishName, rawValue]
            .map(AppLanguage.searchFolded)
            .contains { $0.range(of: needle, options: .literal) != nil }
    }

    public static func searchFolded(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
            .folding(options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive], locale: nil)
    }
}
