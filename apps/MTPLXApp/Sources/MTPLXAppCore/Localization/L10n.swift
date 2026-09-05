import Foundation

// MARK: - L10n
//
// String lookup for the selected app language. The whole UI routes its
// user-facing text through `tr(_:_:)`, which resolves against the
// selected language's `.lproj` directory rather than the process
// language, so switching languages inside the app re-renders without a
// relaunch and unit tests can resolve any language on demand.
//
// Each `<code>.lproj` directory is opened as its own `Bundle`, so the
// lookup never depends on `Bundle.main.preferredLocalizations` or on
// the macOS system language. A key missing from a table (or a table
// missing altogether) returns the key itself, and keys are the English
// source text, so the fallback is always English.
//
// The tables live in `Sources/MTPLXAppCore/Resources/Localization/` in
// the source tree and are copied into `Contents/Resources/<code>.lproj`
// by `script/build_and_run.sh`. Debug builds (and therefore `swift test`)
// also look at the source tree so the tables resolve without a bundle.

public enum L10n {
    public static let tableName = "Localizable"

    /// The language every `tr` call resolves against. `LanguageStore`
    /// keeps this in sync with the persisted preference.
    public static var language: AppLanguage {
        state.withLock { $0.language }
    }

    /// Switches the active language for every subsequent lookup.
    public static func activate(_ language: AppLanguage) {
        state.withLock { $0.language = language }
    }

    /// Resolves `key` in the active language.
    public static func string(_ key: String, arguments: [CVarArg] = []) -> String {
        string(key, language: language, arguments: arguments)
    }

    /// Resolves `key` in an explicit language. Format arguments are
    /// applied with that language's locale so decimal separators and
    /// digit grouping follow the language, not the process locale.
    public static func string(_ key: String, language: AppLanguage, arguments: [CVarArg] = []) -> String {
        let resolved = bundle(for: language)?.localizedString(forKey: key, value: key, table: tableName) ?? key
        guard !arguments.isEmpty else { return resolved }
        return String(format: resolved, locale: language.locale, arguments: arguments)
    }

    /// The `.lproj` directory for `language` opened as a bundle, or nil
    /// when no table ships for it (lookups then fall back to the key).
    public static func bundle(for language: AppLanguage) -> Bundle? {
        if let cached = state.withLock({ $0.bundles[language] }) {
            return cached.bundle
        }
        let located = localizationRoots()
            .map { $0.appendingPathComponent("\(language.code).lproj", isDirectory: true) }
            .first { FileManager.default.fileExists(atPath: $0.appendingPathComponent("\(tableName).strings").path) }
            .flatMap { Bundle(url: $0) }
        state.withLock { $0.bundles[language] = CachedBundle(bundle: located) }
        return located
    }

    /// Directories that may contain `<code>.lproj` tables, in lookup
    /// order: the app bundle's resources first, then (debug builds only)
    /// the source tree so `swift test` and `swift run` resolve the same
    /// tables the packaged app ships.
    public static func localizationRoots() -> [URL] {
        var roots: [URL] = []
        if let resources = Bundle.main.resourceURL {
            roots.append(resources)
        }
        #if DEBUG
        let sourceTree = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Localization
            .deletingLastPathComponent() // MTPLXAppCore
            .appendingPathComponent("Resources", isDirectory: true)
            .appendingPathComponent("Localization", isDirectory: true)
        roots.append(sourceTree)
        #endif
        return roots
    }

    /// Forgets cached bundles so a changed table is re-read. Tests use it;
    /// the app never needs it because tables are immutable at runtime.
    public static func resetBundleCache() {
        state.withLock { $0.bundles.removeAll() }
    }

    // MARK: Storage

    private struct CachedBundle {
        let bundle: Bundle?
    }

    private struct State {
        var language: AppLanguage = .base
        var bundles: [AppLanguage: CachedBundle] = [:]
    }

    private final class Locked<Value>: @unchecked Sendable {
        private var value: Value
        private let lock = NSLock()

        init(_ value: Value) {
            self.value = value
        }

        func withLock<R>(_ body: (inout Value) throws -> R) rethrows -> R {
            lock.lock()
            defer { lock.unlock() }
            return try body(&value)
        }
    }

    private static let state = Locked(State())
}

/// Localized text for the active app language. The key is the English
/// source string; format specifiers (`%@`, `%lld`, `%.1f`) are filled from
/// `arguments` using the language's locale.
///
///     Text(tr("Start MTPLX"))
///     Text(tr("Step %lld of %lld", index + 1, count))
public func tr(_ key: String, _ arguments: CVarArg...) -> String {
    L10n.string(key, arguments: arguments)
}
