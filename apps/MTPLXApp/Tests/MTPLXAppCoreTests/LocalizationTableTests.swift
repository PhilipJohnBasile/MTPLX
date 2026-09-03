import XCTest
import Foundation
@testable import MTPLXAppCore

/// Guards the shipped string tables: every language carries every English
/// key with the same format placeholders, the code only uses keys the
/// English table knows, the onboarding and settings screens are really
/// translated, and the live-switch lookup returns per-language text.
final class LocalizationTableTests: XCTestCase {
    private static let languages = AppLanguage.allCases

    /// The source-tree Localization directory (the DEBUG root of L10n).
    private static var localizationDirectory: URL {
        let candidates = L10n.localizationRoots()
        return candidates.last(where: {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("en.lproj/Localizable.strings").path)
        }) ?? candidates.last!
    }

    private static var packageRoot: URL {
        localizationDirectory  // .../Sources/MTPLXAppCore/Resources/Localization
            .deletingLastPathComponent() // Resources
            .deletingLastPathComponent() // MTPLXAppCore
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // package root
    }

    private static func table(_ language: AppLanguage) throws -> [String: String] {
        let url = localizationDirectory.appendingPathComponent("\(language.code).lproj/Localizable.strings")
        let data = try Data(contentsOf: url)
        let plist = try PropertyListSerialization.propertyList(from: data, options: [], format: nil)
        return try XCTUnwrap(plist as? [String: String], "\(language.code) table must parse as key/value strings")
    }

    private static let specifier = try! NSRegularExpression(
        pattern: #"%(?:\d+\$)?[-+ #0]*\d*(?:\.\d+)?(lld|ld|d|@|lf|f|s|u|x|X|%)"#
    )

    private static func placeholders(_ s: String) -> [String] {
        let ns = s as NSString
        return specifier.matches(in: s, range: NSRange(location: 0, length: ns.length))
            .map { ns.substring(with: $0.range(at: 1)) }
            .filter { $0 != "%" }
            .sorted()
    }

    override func setUp() {
        super.setUp()
        L10n.resetBundleCache()
        L10n.activate(.english)
    }

    override func tearDown() {
        L10n.activate(.english)
        super.tearDown()
    }

    // MARK: Completeness

    func testEveryLanguageShipsATable() throws {
        for language in Self.languages {
            let table = try Self.table(language)
            XCTAssertGreaterThan(table.count, 1000, "\(language.code) table is suspiciously small")
        }
    }

    func testEveryLanguageCarriesExactlyTheEnglishKeys() throws {
        let english = Set(try Self.table(.english).keys)
        for language in Self.languages where language != .english {
            let keys = Set(try Self.table(language).keys)
            let missing = english.subtracting(keys).sorted()
            let extra = keys.subtracting(english).sorted()
            XCTAssertTrue(missing.isEmpty, "\(language.code) is missing \(missing.count) keys, e.g. \(missing.prefix(5))")
            XCTAssertTrue(extra.isEmpty, "\(language.code) has \(extra.count) keys unknown to English, e.g. \(extra.prefix(5))")
        }
    }

    func testEnglishTableMapsEveryKeyToItself() throws {
        for (key, value) in try Self.table(.english) {
            XCTAssertEqual(key, value)
        }
    }

    // MARK: Placeholders

    func testTranslationsPreserveFormatPlaceholders() throws {
        let english = try Self.table(.english)
        for language in Self.languages where language != .english {
            let table = try Self.table(language)
            for (key, value) in table {
                guard english[key] != nil else { continue }
                XCTAssertEqual(
                    Self.placeholders(value), Self.placeholders(key),
                    "\(language.code): placeholder set differs for \(key.debugDescription) -> \(value.debugDescription)"
                )
                XCTAssertFalse(value.trimmingCharacters(in: .whitespaces).isEmpty, "\(language.code): empty value for \(key.debugDescription)")
                XCTAssertEqual(key.hasPrefix(" "), value.hasPrefix(" "), "\(language.code): leading space lost for \(key.debugDescription)")
                XCTAssertEqual(key.hasSuffix(" "), value.hasSuffix(" "), "\(language.code): trailing space lost for \(key.debugDescription)")
            }
        }
    }

    func testPositionalPlaceholdersAreAllOrNothing() throws {
        let positional = try NSRegularExpression(pattern: #"%(\d+)\$"#)
        for language in Self.languages where language != .english {
            for (key, value) in try Self.table(language) {
                let ns = value as NSString
                let positions = positional.matches(in: value, range: NSRange(location: 0, length: ns.length)).count
                guard positions > 0 else { continue }
                XCTAssertEqual(positions, Self.placeholders(key).count, "\(language.code): mixed positional and plain placeholders in \(value.debugDescription)")
            }
        }
    }

    // MARK: Source keys

    func testEverySourceKeyExistsInTheEnglishTable() throws {
        let english = try Self.table(.english)
        let sources = Self.packageRoot.appendingPathComponent("Sources")
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
        let call = try NSRegularExpression(pattern: #"\btr\(\s*"((?:[^"\\]|\\.)*)""#)
        var scanned = 0
        var unknown: [String] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let text = try String(contentsOf: url, encoding: .utf8)
            for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
                let line = String(line)
                if line.trimmingCharacters(in: .whitespaces).hasPrefix("//") { continue }
                let ns = line as NSString
                for match in call.matches(in: line, range: NSRange(location: 0, length: ns.length)) {
                    let key = Self.unescapeSwiftLiteral(ns.substring(with: match.range(at: 1)))
                    scanned += 1
                    if english[key] == nil { unknown.append(key) }
                }
            }
        }
        XCTAssertGreaterThan(scanned, 1500, "expected the tr() call sites to be scanned")
        XCTAssertTrue(unknown.isEmpty, "keys used in code but missing from en.lproj: \(unknown.prefix(10))")
    }

    private static func unescapeSwiftLiteral(_ s: String) -> String {
        var out = ""
        var iterator = s.makeIterator()
        while let c = iterator.next() {
            guard c == "\\" else { out.append(c); continue }
            guard let n = iterator.next() else { break }
            switch n {
            case "n": out.append("\n")
            case "t": out.append("\t")
            case "r": out.append("\r")
            case "0": out.append("\0")
            case "u":
                var hex = ""
                _ = iterator.next() // {
                while let h = iterator.next(), h != "}" { hex.append(h) }
                if let scalar = UInt32(hex, radix: 16).flatMap(Unicode.Scalar.init) { out.unicodeScalars.append(scalar) }
            default: out.append(n)
            }
        }
        return out
    }

    // MARK: No English leftovers on the key screens

    private static let spotKeys = [
        "Choose your language", "Search languages", "Get Started", "Your Mac", "Recommended models",
        "Setting up MTPLX", "Settings", "Performance", "Language", "Cancel", "Continue", "Back",
        "Start MTPLX", "Stop MTPLX", "Done", "Next", "Retry", "Restart", "Clear All",
        "The fastest way to run local AI.", "Speed and batching. Needs a restart to apply.",
        "The language MTPLX uses across the app. Changes apply immediately.",
    ]

    func testOnboardingAndSettingsKeysAreTranslated() throws {
        let english = try Self.table(.english)
        for key in Self.spotKeys {
            XCTAssertNotNil(english[key], "spot key missing from English: \(key)")
        }
        for language in [AppLanguage.simplifiedChinese, .japanese, .arabic, .russian, .hindi, .korean] {
            let table = try Self.table(language)
            for key in Self.spotKeys {
                let value = table[key] ?? ""
                XCTAssertFalse(value.isEmpty, "\(language.code): \(key) untranslated (missing)")
                // Brand tokens may survive inside the value but the value must not be the English text.
                XCTAssertNotEqual(value, key, "\(language.code): \(key) is still English")
            }
        }
    }

    func testGlossaryCancelIsConsistentAcrossLanguages() throws {
        let expected: [AppLanguage: String] = [
            .english: "Cancel", .simplifiedChinese: "取消", .spanish: "Cancelar", .hindi: "रद्द करें",
            .arabic: "إلغاء", .brazilianPortuguese: "Cancelar", .french: "Annuler", .russian: "Отмена",
            .japanese: "キャンセル", .german: "Abbrechen", .korean: "취소", .indonesian: "Batal",
        ]
        for (language, word) in expected {
            XCTAssertEqual(try Self.table(language)["Cancel"], word, language.code)
        }
    }

    // MARK: Live switch

    func testSameKeyResolvesDifferentlyPerLanguage() throws {
        let english = L10n.string("Settings", language: .english)
        XCTAssertEqual(english, "Settings")
        var seen: Set<String> = [english]
        for language in Self.languages where language != .english {
            let value = L10n.string("Settings", language: language)
            XCTAssertNotEqual(value, "Settings", language.code)
            seen.insert(value)
        }
        XCTAssertGreaterThan(seen.count, 6, "translations must not collapse into one string")
    }

    func testActivatingALanguageSwitchesTrImmediately() throws {
        XCTAssertEqual(tr("Cancel"), "Cancel")
        L10n.activate(.japanese)
        XCTAssertEqual(tr("Cancel"), "キャンセル")
        L10n.activate(.german)
        XCTAssertEqual(tr("Cancel"), "Abbrechen")
        L10n.activate(.english)
        XCTAssertEqual(tr("Cancel"), "Cancel")
    }

    @MainActor
    func testLanguageStoreDrivesTr() throws {
        let suite = "mtplx.tests.language.switch.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        addTeardownBlock { UserDefaults(suiteName: suite)?.removePersistentDomain(forName: suite) }
        let store = LanguageStore(defaults: defaults, preferredLanguages: ["en"])
        XCTAssertEqual(tr("Cancel"), "Cancel")
        store.language = .french
        XCTAssertEqual(tr("Cancel"), "Annuler", "tr answers in the new language before any view re-renders")
        store.language = .english
        XCTAssertEqual(tr("Cancel"), "Cancel")
    }

    func testFormatArgumentsFlowThroughTranslatedTemplates() throws {
        let key = "Step %lld of %lld"
        XCTAssertNotNil(try Self.table(.english)[key])
        for language in Self.languages {
            let text = L10n.string(key, language: language, arguments: [2, 7])
            XCTAssertTrue(text.contains("2") && text.contains("7"), "\(language.code): \(text)")
            XCTAssertFalse(text.contains("%lld"), "\(language.code): unformatted placeholder in \(text)")
        }
    }

    // MARK: Bundle script parity

    func testBundleScriptShipsEveryLanguage() throws {
        let script = Self.packageRoot.appendingPathComponent("script/build_and_run.sh")
        let text = try String(contentsOf: script, encoding: .utf8)
        let regex = try NSRegularExpression(pattern: #"LOCALIZATION_CODES=\(([^)]*)\)"#)
        let ns = text as NSString
        let match = try XCTUnwrap(regex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)), "build_and_run.sh must list LOCALIZATION_CODES")
        let codes = ns.substring(with: match.range(at: 1)).split(separator: " ").map(String.init)
        XCTAssertEqual(codes, Self.languages.map(\.code))
    }
}
