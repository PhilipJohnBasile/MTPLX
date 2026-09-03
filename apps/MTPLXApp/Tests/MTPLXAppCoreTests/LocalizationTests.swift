import XCTest
import Foundation
@testable import MTPLXAppCore

/// Language resolution, picker search and the live-switch lookup path.
/// Table completeness and placeholder parity live in
/// `LocalizationTableTests`.
final class LocalizationTests: XCTestCase {
    override func setUp() {
        super.setUp()
        L10n.activate(.english)
    }

    override func tearDown() {
        L10n.activate(.english)
        super.tearDown()
    }

    // MARK: Default language from macOS preferred languages

    func testBestMatchPrefersExactCodeInPreferredOrder() {
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["fr", "en"]), .french)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["zh-Hans", "en"]), .simplifiedChinese)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["pt-BR"]), .brazilianPortuguese)
    }

    func testBestMatchIgnoresRegionAndFillsInScript() {
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["en-GB"]), .english)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["es-419"]), .spanish)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["pt-PT"]), .brazilianPortuguese)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["zh-CN"]), .simplifiedChinese)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["zh-Hans-SG"]), .simplifiedChinese)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["ar-EG"]), .arabic)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["hi-IN"]), .hindi)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["id-ID"]), .indonesian)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["ko-KR"]), .korean)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["ja-JP"]), .japanese)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["ru-RU"]), .russian)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["de-AT"]), .german)
    }

    func testBestMatchDoesNotCrossScripts() {
        // Traditional Chinese is not Simplified Chinese; fall through to
        // the next preferred language exactly like Bundle localization.
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["zh-Hant-TW", "ja"]), .japanese)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["zh-TW"]), .english)
    }

    func testBestMatchFallsBackToEnglish() {
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: []), .english)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["xx-YY", "tlh"]), .english)
        XCTAssertEqual(AppLanguage.bestMatch(preferredLanguages: ["it-IT", "de-DE"]), .german)
    }

    func testCodeInitIsExactAndCaseInsensitive() {
        XCTAssertEqual(AppLanguage(code: "zh-hans"), .simplifiedChinese)
        XCTAssertEqual(AppLanguage(code: " PT-br "), .brazilianPortuguese)
        XCTAssertNil(AppLanguage(code: "zh"))
        XCTAssertNil(AppLanguage(code: ""))
    }

    func testEveryLanguageHasDistinctPresentation() {
        let all = AppLanguage.allCases
        XCTAssertEqual(all.count, 12)
        XCTAssertEqual(Set(all.map(\.code)).count, all.count)
        XCTAssertEqual(Set(all.map(\.nativeName)).count, all.count)
        XCTAssertEqual(Set(all.map(\.englishName)).count, all.count)
        XCTAssertEqual(Set(all.map(\.flag)).count, all.count)
        XCTAssertEqual(all.filter(\.isRightToLeft), [.arabic])
        for language in all {
            XCTAssertFalse(language.flag.isEmpty, language.code)
            XCTAssertEqual(language.locale.identifier, language.code)
        }
    }

    // MARK: Picker search

    func testMatchingEmptyQueryKeepsDeclarationOrder() {
        XCTAssertEqual(AppLanguage.matching(""), AppLanguage.allCases)
        XCTAssertEqual(AppLanguage.matching("   "), AppLanguage.allCases)
    }

    func testMatchingSearchesNativeNameEnglishNameAndCode() {
        XCTAssertEqual(AppLanguage.matching("中文"), [.simplifiedChinese])
        XCTAssertEqual(AppLanguage.matching("Chinese"), [.simplifiedChinese])
        XCTAssertEqual(AppLanguage.matching("zh"), [.simplifiedChinese])
        XCTAssertEqual(AppLanguage.matching("हिन"), [.hindi])
        XCTAssertEqual(AppLanguage.matching("العربية"), [.arabic])
        XCTAssertEqual(AppLanguage.matching("Bahasa"), [.indonesian])
        XCTAssertEqual(AppLanguage.matching("한국"), [.korean])
        XCTAssertEqual(AppLanguage.matching("Русский"), [.russian])
    }

    func testMatchingIsCaseAndDiacriticInsensitive() {
        XCTAssertEqual(AppLanguage.matching("ESPANOL"), [.spanish])
        XCTAssertEqual(AppLanguage.matching("français"), [.french])
        XCTAssertEqual(AppLanguage.matching("FRANCAIS"), [.french])
        XCTAssertEqual(AppLanguage.matching("portugues"), [.brazilianPortuguese])
        XCTAssertEqual(AppLanguage.matching("русский"), [.russian])
    }

    func testMatchingTrimsAndReturnsNothingForGarbage() {
        XCTAssertEqual(AppLanguage.matching("  japan  "), [.japanese])
        XCTAssertEqual(AppLanguage.matching("zzzz"), [])
    }

    // MARK: Store persistence

    private func scratchDefaults() throws -> UserDefaults {
        let suite = "mtplx.tests.language.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        addTeardownBlock {
            UserDefaults(suiteName: suite)?.removePersistentDomain(forName: suite)
        }
        return defaults
    }

    @MainActor
    func testStoreDefaultsToPreferredLanguageWhenNothingIsPersisted() throws {
        let defaults = try scratchDefaults()
        let store = LanguageStore(defaults: defaults, preferredLanguages: ["ko-KR", "en"])
        XCTAssertEqual(store.language, .korean)
        XCTAssertFalse(store.isPersisted)
        XCTAssertEqual(L10n.language, .korean, "constructing the store activates the language")
    }

    @MainActor
    func testStorePersistsUnderTheAppStorageKeyAndReactivates() throws {
        let defaults = try scratchDefaults()
        let store = LanguageStore(defaults: defaults, preferredLanguages: ["en"])
        store.language = .japanese
        XCTAssertEqual(defaults.string(forKey: "mtplx.app.language"), "ja")
        XCTAssertEqual(LanguageStore.defaultsKey, "mtplx.app.language")
        XCTAssertTrue(store.isPersisted)
        XCTAssertEqual(L10n.language, .japanese)

        let relaunched = LanguageStore(defaults: defaults, preferredLanguages: ["de"])
        XCTAssertEqual(relaunched.language, .japanese, "persisted choice beats the system preference")
    }

    @MainActor
    func testStoreIgnoresUnknownPersistedCode() throws {
        let defaults = try scratchDefaults()
        defaults.set("tlh", forKey: LanguageStore.defaultsKey)
        let store = LanguageStore(defaults: defaults, preferredLanguages: ["es"])
        XCTAssertEqual(store.language, .spanish)
    }

    // MARK: Lookup mechanism

    func testMissingKeyFallsBackToTheEnglishKeyItself() {
        let key = "This key does not exist in any table \(UUID().uuidString)"
        for language in AppLanguage.allCases {
            XCTAssertEqual(L10n.string(key, language: language), key, language.code)
        }
    }

    func testFormatArgumentsUseTheLanguageLocale() {
        let key = "Unit test format %.1f \(UUID().uuidString)"
        XCTAssertEqual(L10n.string(key, language: .english, arguments: [1.5]), "Unit test format 1.5 " + String(key.dropFirst("Unit test format %.1f ".count)))
        XCTAssertTrue(L10n.string(key, language: .german, arguments: [1.5]).contains("1,5"), "German decimal comma")
    }

    func testTrFollowsTheActiveLanguage() {
        L10n.activate(.french)
        XCTAssertEqual(L10n.language, .french)
        let key = "Untranslated \(UUID().uuidString)"
        XCTAssertEqual(tr(key), key)
        L10n.activate(.english)
        XCTAssertEqual(L10n.language, .english)
    }
}
