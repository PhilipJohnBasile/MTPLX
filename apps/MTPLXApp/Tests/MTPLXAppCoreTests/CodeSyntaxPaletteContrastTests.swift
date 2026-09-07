import AppKit
import XCTest

@testable import MTPLXAppCore

/// The code viewport palettes must hold WCAG AA (4.5:1) on the surface
/// they are drawn over, in both looks, and the adaptive palette the lexer
/// emits must resolve to the right set from the drawing appearance.
final class CodeSyntaxPaletteContrastTests: XCTestCase {
    private typealias Palette = MTPLXCodeHighlighter.Palette

    /// `Brand.bgInner` per look, the fence surface the code viewports sit
    /// on. The app target's BrandPaletteTests assert the same two values
    /// so the surfaces cannot drift apart silently.
    private static let jetCodeSurface: UInt32 = 0x121212
    private static let creamCodeSurface: UInt32 = 0xF9F5EB

    private static func roles(of palette: Palette) -> [(name: String, color: NSColor)] {
        [
            ("base", palette.base),
            ("keyword", palette.keyword),
            ("type", palette.type),
            ("string", palette.string),
            ("comment", palette.comment),
            ("number", palette.number),
            ("function", palette.function),
            ("decorator", palette.decorator),
        ]
    }

    // MARK: - WCAG 2.x math

    private static func linear(_ c: Double) -> Double {
        c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
    }

    private static func luminance(_ hex: UInt32) -> Double {
        0.2126 * linear(Double((hex >> 16) & 0xFF) / 255.0)
            + 0.7152 * linear(Double((hex >> 8) & 0xFF) / 255.0)
            + 0.0722 * linear(Double(hex & 0xFF) / 255.0)
    }

    private static func luminance(_ color: NSColor) -> Double? {
        guard let srgb = color.usingColorSpace(.sRGB) else { return nil }
        return 0.2126 * linear(srgb.redComponent)
            + 0.7152 * linear(srgb.greenComponent)
            + 0.0722 * linear(srgb.blueComponent)
    }

    private static func contrast(_ a: Double, _ b: Double) -> Double {
        (max(a, b) + 0.05) / (min(a, b) + 0.05)
    }

    private func assertPaletteMeetsAA(_ palette: Palette, named paletteName: String, on surface: UInt32) throws {
        let surfaceLuminance = Self.luminance(surface)
        for role in Self.roles(of: palette) {
            let roleLuminance = try XCTUnwrap(Self.luminance(role.color), "\(paletteName).\(role.name) is not convertible to sRGB")
            let ratio = Self.contrast(roleLuminance, surfaceLuminance)
            XCTAssertGreaterThanOrEqual(
                ratio, 4.5,
                String(format: "%@.%@ is %.2f:1 on 0x%06X, needs 4.5:1", paletteName, role.name, ratio, surface)
            )
        }
    }

    func testDarkPaletteMeetsAAOnTheJetCodeSurface() throws {
        try assertPaletteMeetsAA(.dark, named: "dark", on: Self.jetCodeSurface)
    }

    func testLightPaletteMeetsAAOnTheCreamCodeSurface() throws {
        try assertPaletteMeetsAA(.light, named: "light", on: Self.creamCodeSurface)
    }

    func testLightPaletteIsDarkInkOnPaper() throws {
        // Every light role must be darker than the mid-grey the dark set
        // sits above, so a copy-paste of the dark values cannot pass.
        for role in Self.roles(of: .light) {
            let luminance = try XCTUnwrap(Self.luminance(role.color))
            XCTAssertLessThan(luminance, 0.2, "light.\(role.name) should read as ink on paper")
        }
        for role in Self.roles(of: .dark) {
            let luminance = try XCTUnwrap(Self.luminance(role.color))
            XCTAssertGreaterThan(luminance, 0.2, "dark.\(role.name) should read as light on jet")
        }
    }

    // MARK: - Adaptive resolution

    private func components(_ color: NSColor, under appearanceName: NSAppearance.Name) throws -> [Double] {
        let appearance = try XCTUnwrap(NSAppearance(named: appearanceName))
        var resolved: NSColor?
        appearance.performAsCurrentDrawingAppearance {
            resolved = color.usingColorSpace(.sRGB)
        }
        let srgb = try XCTUnwrap(resolved)
        return [srgb.redComponent, srgb.greenComponent, srgb.blueComponent, srgb.alphaComponent]
    }

    func testAdaptivePaletteResolvesToDarkAndLightByAppearance() throws {
        let adaptive = Self.roles(of: .adaptive)
        let dark = Self.roles(of: .dark)
        let light = Self.roles(of: .light)
        for index in adaptive.indices {
            let role = adaptive[index].name
            let underDark = try components(adaptive[index].color, under: .darkAqua)
            let expectedDark = try components(dark[index].color, under: .darkAqua)
            let underLight = try components(adaptive[index].color, under: .aqua)
            let expectedLight = try components(light[index].color, under: .aqua)
            for channel in 0..<4 {
                XCTAssertEqual(underDark[channel], expectedDark[channel], accuracy: 0.002, "adaptive.\(role) under darkAqua should be the dark tone")
                XCTAssertEqual(underLight[channel], expectedLight[channel], accuracy: 0.002, "adaptive.\(role) under aqua should be the light tone")
            }
        }
    }

    func testLexerEmitsTheAdaptivePalette() {
        let (line, _) = MTPLXCodeHighlighter.highlightLine(
            "def go(x=3):  # start",
            language: .python,
            state: .none
        )
        var emitted: [NSColor] = []
        line.enumerateAttribute(.foregroundColor, in: NSRange(location: 0, length: line.length)) { value, _, _ in
            if let color = value as? NSColor { emitted.append(color) }
        }
        XCTAssertFalse(emitted.isEmpty)
        let adaptive = Self.roles(of: .adaptive).map(\.color)
        for color in emitted {
            XCTAssertTrue(adaptive.contains { $0 === color }, "lexer emitted a color outside Palette.adaptive")
        }
    }
}
