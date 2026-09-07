import Foundation
import Combine

// MARK: - LanguageStore
//
// The user's language choice. Persisted in `UserDefaults` under
// `mtplx.app.language` (the same key an `@AppStorage` reader would use)
// as the BCP 47 code, next to the other `mtplx.app.*` preferences.
//
// First launch resolves the default from the user's macOS preferred
// languages (best match, else English). Every write activates the
// language in `L10n` before publishing, so by the time views re-render
// every `tr` call already answers in the new language.

@MainActor
public final class LanguageStore: ObservableObject {
    public static let defaultsKey = "mtplx.app.language"

    @Published public var language: AppLanguage {
        didSet {
            guard language != oldValue else { return }
            L10n.activate(language)
            defaults.set(language.code, forKey: Self.defaultsKey)
        }
    }

    private let defaults: UserDefaults

    public init(
        defaults: UserDefaults = .standard,
        preferredLanguages: [String] = Locale.preferredLanguages
    ) {
        self.defaults = defaults
        let stored = defaults.string(forKey: Self.defaultsKey).flatMap(AppLanguage.init(code:))
        let resolved = stored ?? AppLanguage.bestMatch(preferredLanguages: preferredLanguages)
        language = resolved
        L10n.activate(resolved)
    }

    /// True once the user (or a previous launch) has chosen explicitly;
    /// false while the language is still the preferred-languages default.
    public var isPersisted: Bool {
        defaults.string(forKey: Self.defaultsKey) != nil
    }
}
