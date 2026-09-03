import AppKit
import SwiftUI
import XCTest

@testable import MTPLXAppHost

/// Guards the two brand palettes behind the appearance preference.
///
/// 1. The dark palette is the shipped Jet Chrome identity and must stay
///    byte-identical: light mode was added by making the tokens dynamic,
///    so a stray tweak here would silently restyle the default look.
/// 2. Every type-on-surface pair meets WCAG AA in both palettes (4.5:1
///    for the body tiers and semantic tones, 3:1 for the quietest tier
///    and the large chrome gradients), computed from the token values so
///    a future tone change cannot regress either mode.
/// 3. The dynamic tokens really do resolve per appearance, through both
///    the AppKit and the SwiftUI resolution paths the app uses.
final class BrandPaletteTests: XCTestCase {
    private typealias Tone = BrandPalette.Tone
    private typealias ToneKey = KeyPath<BrandPalette, Tone>
    private typealias Named = (name: String, key: ToneKey)

    private static let palettes: [(name: String, palette: BrandPalette)] = [
        ("dark", .dark),
        ("light", .light),
    ]

    private static var surfaces: [Named] {
        [
            ("bgOuter", \.bgOuter),
            ("bgMid", \.bgMid),
            ("bgInner", \.bgInner),
            ("cardSurface", \.cardSurface),
            ("raisedSurface", \.raisedSurface),
            ("panelSurfaceTop", \.panelSurfaceTop),
            ("panelSurfaceBottom", \.panelSurfaceBottom),
            ("floorCenter", \.floorCenter),
            ("floorEdge", \.floorEdge),
        ]
    }

    private static var bodyType: [Named] {
        [
            ("typeHi", \.typeHi),
            ("typeBody", \.typeBody),
            ("typeSecondary", \.typeSecondary),
        ]
    }

    private static var quietType: [Named] {
        [
            ("typeTertiary", \.typeTertiary),
        ]
    }

    private static var semanticTones: [Named] {
        [
            ("warning", \.warning),
            ("danger", \.danger),
            ("success", \.success),
            ("accentChrome", \.accentChrome),
            ("accentWarm", \.accentWarm),
            ("coolChrome", \.coolChrome),
        ]
    }

    private static var gradientStops: [Named] {
        [
            ("chromeStop0", \.chromeStop0),
            ("chromeStop1", \.chromeStop1),
            ("chromeStop2", \.chromeStop2),
            ("chromeStop3", \.chromeStop3),
            ("chromeStop4", \.chromeStop4),
            ("typeGradientTop", \.typeGradientTop),
            ("typeGradientMid", \.typeGradientMid),
            ("typeGradientBottom", \.typeGradientBottom),
        ]
    }

    // MARK: - WCAG 2.x math

    private static func channel(_ value: UInt32) -> Double {
        let c = Double(value) / 255.0
        return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
    }

    static func relativeLuminance(_ hex: UInt32) -> Double {
        0.2126 * channel((hex >> 16) & 0xFF)
            + 0.7152 * channel((hex >> 8) & 0xFF)
            + 0.0722 * channel(hex & 0xFF)
    }

    static func contrastRatio(_ a: UInt32, _ b: UInt32) -> Double {
        let la = relativeLuminance(a)
        let lb = relativeLuminance(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    private func assertContrast(
        _ foreground: Named,
        on background: Named,
        in palette: (name: String, palette: BrandPalette),
        atLeast floor: Double,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let fg = palette.palette[keyPath: foreground.key]
        let bg = palette.palette[keyPath: background.key]
        XCTAssertEqual(fg.alpha, 1, "\(palette.name).\(foreground.name) must be opaque", file: file, line: line)
        XCTAssertEqual(bg.alpha, 1, "\(palette.name).\(background.name) must be opaque", file: file, line: line)
        let ratio = Self.contrastRatio(fg.hex, bg.hex)
        XCTAssertGreaterThanOrEqual(
            ratio,
            floor,
            String(
                format: "%@: %@ on %@ is %.2f:1, needs %.1f:1",
                palette.name, foreground.name, background.name, ratio, floor
            ),
            file: file,
            line: line
        )
    }

    private func hexString(_ tone: Tone) -> String {
        String(format: "0x%06X @ %.2f", tone.hex, tone.alpha)
    }

    // MARK: - Dark palette is frozen

    func testDarkPaletteIsTheShippedJetChrome() {
        let dark = BrandPalette.dark
        let expected: [(String, Tone, Tone)] = [
            ("bgInner", dark.bgInner, Tone(0x121212)),
            ("bgMid", dark.bgMid, Tone(0x0A0A0A)),
            ("bgOuter", dark.bgOuter, Tone(0x050505)),
            ("cardSurface", dark.cardSurface, Tone(0x101010)),
            ("raisedSurface", dark.raisedSurface, Tone(0x161616)),
            ("panelSurfaceTop", dark.panelSurfaceTop, Tone(0x1A1A1A)),
            ("panelSurfaceBottom", dark.panelSurfaceBottom, Tone(0x101010)),
            ("floorCenter", dark.floorCenter, Tone(0x050505)),
            ("floorEdge", dark.floorEdge, Tone(0x050505)),
            ("typeHi", dark.typeHi, Tone(0xEFEFEF)),
            ("typeBody", dark.typeBody, Tone(0xDEDEDE)),
            ("typeSecondary", dark.typeSecondary, Tone(0x9A9A9A)),
            ("typeTertiary", dark.typeTertiary, Tone(0x6A6A6A)),
            ("invertedType", dark.invertedType, Tone(0x000000)),
            ("typeGradientTop", dark.typeGradientTop, Tone(0xFFFFFF)),
            ("typeGradientMid", dark.typeGradientMid, Tone(0xF0F0EA)),
            ("typeGradientBottom", dark.typeGradientBottom, Tone(0xCFCFC8)),
            ("chromeStop0", dark.chromeStop0, Tone(0xF6F6F6)),
            ("chromeStop1", dark.chromeStop1, Tone(0xE0E0E0)),
            ("chromeStop2", dark.chromeStop2, Tone(0x9A9A9A)),
            ("chromeStop3", dark.chromeStop3, Tone(0xE0E0E0)),
            ("chromeStop4", dark.chromeStop4, Tone(0xF6F6F6)),
            ("accentChrome", dark.accentChrome, Tone(0xC8D0D5)),
            ("accentWarm", dark.accentWarm, Tone(0xD5CFC4)),
            ("coolChrome", dark.coolChrome, Tone(0xBFD4E0)),
            ("chromeHalo", dark.chromeHalo, Tone(0xFFFFFF, alpha: 0.06)),
            ("warning", dark.warning, Tone(0xE9C46A)),
            ("danger", dark.danger, Tone(0xE76F51)),
            ("success", dark.success, Tone(0x88D498)),
            ("separator", dark.separator, Tone(0xFFFFFF, alpha: 0.06)),
            ("separatorStrong", dark.separatorStrong, Tone(0xFFFFFF, alpha: 0.14)),
            ("wash", dark.wash, Tone(0xFFFFFF)),
            ("shade", dark.shade, Tone(0x000000)),
            ("ink", dark.ink, Tone(0x000000)),
            ("onInk", dark.onInk, Tone(0xEFEFEF)),
            ("onAccent", dark.onAccent, Tone(0xFFFFFF)),
            ("hudFill", dark.hudFill, Tone(0x000000, alpha: 0.72)),
            ("shadowLow", dark.shadowLow, Tone(0x000000, alpha: 0.30)),
            ("shadowMid", dark.shadowMid, Tone(0x000000, alpha: 0.40)),
            ("shadowHi", dark.shadowHi, Tone(0x000000, alpha: 0.50)),
        ]
        for (name, actual, frozen) in expected {
            XCTAssertEqual(
                actual, frozen,
                "dark.\(name) drifted from the shipped Jet Chrome value: \(hexString(actual)) vs \(hexString(frozen))"
            )
        }
    }

    func testDarkSurfacesAreTrueNeutral() {
        for surface in Self.surfaces {
            let hex = BrandPalette.dark[keyPath: surface.key].hex
            let r = (hex >> 16) & 0xFF, g = (hex >> 8) & 0xFF, b = hex & 0xFF
            XCTAssertTrue(r == g && g == b, "dark.\(surface.name) must be R == G == B, got \(String(format: "0x%06X", hex))")
        }
    }

    // MARK: - Contrast, both palettes

    func testBodyTypeMeetsAAOnEverySurface() {
        for palette in Self.palettes {
            for surface in Self.surfaces {
                for type in Self.bodyType {
                    assertContrast(type, on: surface, in: palette, atLeast: 4.5)
                }
            }
        }
    }

    func testQuietTypeMeetsTheLargeTextMinimumOnEverySurface() {
        for palette in Self.palettes {
            for surface in Self.surfaces {
                for type in Self.quietType {
                    assertContrast(type, on: surface, in: palette, atLeast: 3.0)
                }
            }
        }
    }

    func testSemanticTonesMeetAAOnEverySurface() {
        for palette in Self.palettes {
            for surface in Self.surfaces {
                for tone in Self.semanticTones {
                    assertContrast(tone, on: surface, in: palette, atLeast: 4.5)
                }
            }
        }
    }

    func testGradientStopsMeetTheLargeTextMinimumOnEverySurface() {
        for palette in Self.palettes {
            for surface in Self.surfaces {
                for stop in Self.gradientStops {
                    assertContrast(stop, on: surface, in: palette, atLeast: 3.0)
                }
            }
        }
    }

    func testInvertedControlsMeetAA() {
        for palette in Self.palettes {
            // Off-white / ink pills with inverted text (MTPLXPillButton,
            // UpdateAvailableToast, the onboarding and forge CTAs).
            assertContrast(("invertedType", \.invertedType), on: ("typeHi", \.typeHi), in: palette, atLeast: 4.5)
            assertContrast(("invertedType", \.invertedType), on: ("typeBody", \.typeBody), in: palette, atLeast: 4.5)
            assertContrast(("bgOuter", \.bgOuter), on: ("typeHi", \.typeHi), in: palette, atLeast: 4.5)
            assertContrast(("bgOuter", \.bgOuter), on: ("typeBody", \.typeBody), in: palette, atLeast: 4.5)
            // Glyphs on ink discs, and the white glyph on the danger pill
            // (a UI component, so the 3:1 non-text bar; the shipped dark
            // pill sits at 3.1:1).
            assertContrast(("onInk", \.onInk), on: ("ink", \.ink), in: palette, atLeast: 4.5)
            assertContrast(("onAccent", \.onAccent), on: ("danger", \.danger), in: palette, atLeast: 3.0)
        }
        // The send glyph on the accent fill. Dark ships a white arrow on
        // light steel (1.6:1) and is frozen by pixel identity; the cream
        // look puts the same white on dark steel and must hold AA.
        assertContrast(("onAccent", \.onAccent), on: ("accentChrome", \.accentChrome), in: Self.palettes[1], atLeast: 4.5)
    }

    // MARK: - Light palette is a curated cream, not an inversion

    func testLightPaletteIsWarmPaperNotAnInversion() {
        let light = BrandPalette.light
        let dark = BrandPalette.dark
        for surface in Self.surfaces {
            let hex = light[keyPath: surface.key].hex
            let r = (hex >> 16) & 0xFF, g = (hex >> 8) & 0xFF, b = hex & 0xFF
            XCTAssertTrue(r >= g && g > b, "light.\(surface.name) should be warm (R >= G > B), got \(String(format: "0x%06X", hex))")
            XCTAssertNotEqual(hex, 0xFFFFFF ^ dark[keyPath: surface.key].hex, "light.\(surface.name) must not be a bit inversion of the dark surface")
        }
        for type in Self.bodyType + Self.quietType {
            let hex = light[keyPath: type.key].hex
            let r = (hex >> 16) & 0xFF, b = hex & 0xFF
            XCTAssertGreaterThan(r, b, "light.\(type.name) should be warm ink, got \(String(format: "0x%06X", hex))")
            XCTAssertNotEqual(hex, 0xFFFFFF ^ dark[keyPath: type.key].hex, "light.\(type.name) must not be a bit inversion of the dark tier")
        }
        // Cards and raised surfaces sit brighter than the floor on paper.
        XCTAssertGreaterThan(Self.relativeLuminance(light.cardSurface.hex), Self.relativeLuminance(light.bgOuter.hex))
        XCTAssertGreaterThan(Self.relativeLuminance(light.raisedSurface.hex), Self.relativeLuminance(light.cardSurface.hex))
        // The vignette is subtle: no white blast, no visible seam.
        XCTAssertGreaterThan(Self.relativeLuminance(light.floorCenter.hex), Self.relativeLuminance(light.floorEdge.hex))
        XCTAssertLessThan(Self.contrastRatio(light.floorCenter.hex, light.floorEdge.hex), 1.15)
        // Shadows are warm and softer than the jet ones.
        XCTAssertLessThan(light.shadowLow.alpha, dark.shadowLow.alpha)
        XCTAssertLessThan(light.shadowHi.alpha, dark.shadowHi.alpha)
        XCTAssertGreaterThan((light.shade.hex >> 16) & 0xFF, light.shade.hex & 0xFF, "light.shade should be a warm umber")
    }

    /// The Core-side CodeSyntaxPaletteContrastTests measure the syntax
    /// hues against these two surfaces; keep them in step.
    func testCodeSurfacesMatchTheCorePaletteTest() {
        XCTAssertEqual(BrandPalette.dark.bgInner.hex, 0x121212)
        XCTAssertEqual(BrandPalette.light.bgInner.hex, 0xF9F5EB)
    }

    // MARK: - Dynamic resolution

    private func srgb(_ color: NSColor, under appearanceName: NSAppearance.Name) -> (r: Double, g: Double, b: Double, a: Double)? {
        guard let appearance = NSAppearance(named: appearanceName) else { return nil }
        var resolved: NSColor?
        appearance.performAsCurrentDrawingAppearance {
            resolved = color.usingColorSpace(.sRGB)
        }
        guard let resolved else { return nil }
        return (
            Double(resolved.redComponent),
            Double(resolved.greenComponent),
            Double(resolved.blueComponent),
            Double(resolved.alphaComponent)
        )
    }

    func testAppKitTokensResolvePerAppearance() throws {
        let floor = BrandPalette.nsColor(\.bgOuter)
        let dark = try XCTUnwrap(srgb(floor, under: .darkAqua))
        let light = try XCTUnwrap(srgb(floor, under: .aqua))
        XCTAssertEqual(dark.r, 0x05 / 255.0, accuracy: 0.002)
        XCTAssertEqual(light.r, 0xF5 / 255.0, accuracy: 0.002)
        XCTAssertEqual(light.b, 0xE1 / 255.0, accuracy: 0.002)

        // High-contrast variants fold onto their base look.
        let highContrastDark = try XCTUnwrap(srgb(floor, under: .accessibilityHighContrastDarkAqua))
        XCTAssertEqual(highContrastDark.r, 0x05 / 255.0, accuracy: 0.002)

        // A tone with its own alpha keeps it through resolution.
        let separator = BrandPalette.nsColor(\.separator)
        let darkSeparator = try XCTUnwrap(srgb(separator, under: .darkAqua))
        let lightSeparator = try XCTUnwrap(srgb(separator, under: .aqua))
        XCTAssertEqual(darkSeparator.a, 0.06, accuracy: 0.002)
        XCTAssertEqual(lightSeparator.a, 0.10, accuracy: 0.002)
    }

    func testSwiftUITokensResolvePerColorScheme() {
        var darkEnvironment = EnvironmentValues()
        darkEnvironment.colorScheme = .dark
        var lightEnvironment = EnvironmentValues()
        lightEnvironment.colorScheme = .light

        let darkFloor = Brand.bgOuter.resolve(in: darkEnvironment)
        let lightFloor = Brand.bgOuter.resolve(in: lightEnvironment)
        XCTAssertEqual(Double(darkFloor.red), 0x05 / 255.0, accuracy: 0.004)
        XCTAssertEqual(Double(lightFloor.red), 0xF5 / 255.0, accuracy: 0.004)

        // Call-site opacities multiply the tone's own alpha, in both looks.
        let darkWash = Brand.wash.opacity(0.04).resolve(in: darkEnvironment)
        let lightWash = Brand.wash.opacity(0.04).resolve(in: lightEnvironment)
        XCTAssertEqual(Double(darkWash.red), 1.0, accuracy: 0.004)
        XCTAssertEqual(Double(darkWash.opacity), 0.04, accuracy: 0.004)
        XCTAssertEqual(Double(lightWash.red), 0x2B / 255.0, accuracy: 0.004)
        XCTAssertEqual(Double(lightWash.opacity), 0.04, accuracy: 0.004)

        let darkShade = Brand.shade.opacity(0.55).resolve(in: darkEnvironment)
        let lightShade = Brand.shade.opacity(0.55).resolve(in: lightEnvironment)
        XCTAssertEqual(Double(darkShade.opacity), 0.55, accuracy: 0.004)
        XCTAssertEqual(Double(lightShade.opacity), 0.36 * 0.55, accuracy: 0.004)
    }

    func testTokensAreSharedInstances() {
        // Static lets: repeated reads hand back the same NSColor, so no
        // token allocates per render.
        XCTAssertTrue(BrandPalette.dark.bgOuter.nsColor === BrandPalette.dark.bgOuter.nsColor)
        XCTAssertTrue(BrandPalette.light.typeHi.nsColor === BrandPalette.light.typeHi.nsColor)
        XCTAssertEqual(Brand.bgOuter, Brand.bgOuter)
    }

    // MARK: - Appearance preference

    func testAppearancePreferenceMapping() {
        XCTAssertEqual(AppAppearance.allCases, [.system, .dark, .light])
        XCTAssertEqual(AppAppearance.system.rawValue, "system")
        XCTAssertEqual(AppAppearance.dark.rawValue, "dark")
        XCTAssertEqual(AppAppearance.light.rawValue, "light")
        XCTAssertNil(AppAppearance.system.colorScheme)
        XCTAssertEqual(AppAppearance.dark.colorScheme, .dark)
        XCTAssertEqual(AppAppearance.light.colorScheme, .light)
        XCTAssertEqual(ThemeStore.appearanceKey, "mtplx.app.appearance")
    }

    @MainActor
    func testAppearancePreferenceDefaultsToDarkAndPersists() {
        let defaults = UserDefaults.standard
        let key = ThemeStore.appearanceKey
        let previous = defaults.object(forKey: key)
        defer {
            if let previous {
                defaults.set(previous, forKey: key)
            } else {
                defaults.removeObject(forKey: key)
            }
        }

        defaults.removeObject(forKey: key)
        XCTAssertEqual(ThemeStore().appearance, .dark, "first launch must keep the Jet Chrome look")

        let store = ThemeStore()
        store.appearance = .light
        XCTAssertEqual(defaults.string(forKey: key), "light")
        XCTAssertEqual(ThemeStore().appearance, .light)

        defaults.set("not-a-look", forKey: key)
        XCTAssertEqual(ThemeStore().appearance, .dark, "an unknown stored value falls back to dark")
    }
}
