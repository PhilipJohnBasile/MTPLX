import SwiftUI
import AppKit

// MARK: - Color hex sugar

extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255.0,
            green: Double((hex >> 8) & 0xFF) / 255.0,
            blue: Double(hex & 0xFF) / 255.0,
            opacity: 1
        )
    }

    init(hex: UInt32, alpha: Double) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255.0,
            green: Double((hex >> 8) & 0xFF) / 255.0,
            blue: Double(hex & 0xFF) / 255.0,
            opacity: alpha
        )
    }
}

// MARK: - BrandPalette
//
// One appearance's worth of MTPLX tones. `dark` is the shipped "Jet
// Chrome" identity, byte for byte: every hex in it is the constant the
// app rendered before light mode existed, and BrandPaletteTests freezes
// those values so a tweak cannot drift the dark look. `light` is the
// curated cream counterpart: warm paper surfaces, warm ink type, a
// graphite take on the polished-steel chrome, and status tones re-tuned
// to hold WCAG AA on cream. It is a designed palette, not an inversion.
//
// `Brand` reads every token through an appearance-aware NSColor, so the
// hundreds of existing `Brand.*` call sites keep compiling and pick the
// right tone from the window's effective appearance. Every tone and its
// NSColor is built once (static lets); resolving a token at draw time is
// a key-path read, never an allocation.

struct BrandPalette: Sendable {
    /// One sRGB tone with an alpha. The NSColor is created when the
    /// palette is built, so the dynamic providers below hand back a
    /// cached instance instead of allocating during a render pass.
    struct Tone: Sendable, Equatable {
        let hex: UInt32
        let alpha: Double
        let nsColor: NSColor

        init(_ hex: UInt32, alpha: Double = 1) {
            self.hex = hex
            self.alpha = alpha
            self.nsColor = NSColor(
                srgbRed: Double((hex >> 16) & 0xFF) / 255.0,
                green: Double((hex >> 8) & 0xFF) / 255.0,
                blue: Double(hex & 0xFF) / 255.0,
                alpha: alpha
            )
        }

        static func == (lhs: Tone, rhs: Tone) -> Bool {
            lhs.hex == rhs.hex && lhs.alpha == rhs.alpha
        }
    }

    // Surfaces
    let bgInner: Tone
    let bgMid: Tone
    let bgOuter: Tone
    let cardSurface: Tone
    let raisedSurface: Tone
    let panelSurfaceTop: Tone
    let panelSurfaceBottom: Tone
    /// Stops of the window floor. Equal in dark (a flat piano black);
    /// a subtle warm vignette in light.
    let floorCenter: Tone
    let floorEdge: Tone

    // Type
    let typeHi: Tone
    let typeBody: Tone
    let typeSecondary: Tone
    let typeTertiary: Tone
    /// Type painted on a `typeHi` / `typeBody` filled pill (the inverted
    /// CTA): black on the off-white pill in dark, paper on the ink pill
    /// in light.
    let invertedType: Tone
    let typeGradientTop: Tone
    let typeGradientMid: Tone
    let typeGradientBottom: Tone

    // Chrome
    let chromeStop0: Tone
    let chromeStop1: Tone
    let chromeStop2: Tone
    let chromeStop3: Tone
    let chromeStop4: Tone
    let accentChrome: Tone
    let accentWarm: Tone
    let coolChrome: Tone
    /// Ambient halo behind chrome text: a faint white glow on jet, a
    /// soft warm shadow on cream.
    let chromeHalo: Tone

    // Status
    let warning: Tone
    let danger: Tone
    let success: Tone

    // Hairlines, washes, shadows
    let separator: Tone
    let separatorStrong: Tone
    /// The tone every chip, hover wash, and bevel edge is built from at
    /// low opacity: white on jet, warm ink on cream.
    let wash: Tone
    /// The tone every drop shadow is built from. Black on jet; on cream
    /// a warm umber pre-dimmed so the same call-site opacity yields a
    /// soft paper shadow instead of a hard grey one.
    let shade: Tone
    /// Solid ink used where a dark layer must stay dark in both looks
    /// (badge glyphs, the attachment remove disc).
    let ink: Tone
    /// Type painted on `ink`.
    let onInk: Tone
    /// Type and glyphs painted on an `accentChrome` or `danger` fill.
    let onAccent: Tone
    /// Backdrop of the floating render HUD.
    let hudFill: Tone
    let shadowLow: Tone
    let shadowMid: Tone
    let shadowHi: Tone

    /// Jet Chrome. True-neutral dark surfaces (R == G == B), off-white
    /// type tiers, polished-steel chrome. Frozen by BrandPaletteTests.
    static let dark = BrandPalette(
        bgInner: Tone(0x121212),
        bgMid: Tone(0x0A0A0A),
        bgOuter: Tone(0x050505),
        cardSurface: Tone(0x101010),
        raisedSurface: Tone(0x161616),
        panelSurfaceTop: Tone(0x1A1A1A),
        panelSurfaceBottom: Tone(0x101010),
        floorCenter: Tone(0x050505),
        floorEdge: Tone(0x050505),
        typeHi: Tone(0xEFEFEF),
        typeBody: Tone(0xDEDEDE),
        typeSecondary: Tone(0x9A9A9A),
        typeTertiary: Tone(0x6A6A6A),
        invertedType: Tone(0x000000),
        typeGradientTop: Tone(0xFFFFFF),
        typeGradientMid: Tone(0xF0F0EA),
        typeGradientBottom: Tone(0xCFCFC8),
        chromeStop0: Tone(0xF6F6F6),
        chromeStop1: Tone(0xE0E0E0),
        chromeStop2: Tone(0x9A9A9A),
        chromeStop3: Tone(0xE0E0E0),
        chromeStop4: Tone(0xF6F6F6),
        accentChrome: Tone(0xC8D0D5),
        accentWarm: Tone(0xD5CFC4),
        coolChrome: Tone(0xBFD4E0),
        chromeHalo: Tone(0xFFFFFF, alpha: 0.06),
        warning: Tone(0xE9C46A),
        danger: Tone(0xE76F51),
        success: Tone(0x88D498),
        separator: Tone(0xFFFFFF, alpha: 0.06),
        separatorStrong: Tone(0xFFFFFF, alpha: 0.14),
        wash: Tone(0xFFFFFF),
        shade: Tone(0x000000),
        ink: Tone(0x000000),
        onInk: Tone(0xEFEFEF),
        onAccent: Tone(0xFFFFFF),
        hudFill: Tone(0x000000, alpha: 0.72),
        shadowLow: Tone(0x000000, alpha: 0.30),
        shadowMid: Tone(0x000000, alpha: 0.40),
        shadowHi: Tone(0x000000, alpha: 0.50)
    )

    /// Cream. Warm paper floor with brighter cream cards and raised
    /// surfaces, warm near-black ink type, graphite chrome that still
    /// reads as light on a bevel, and warm hairlines and shadows.
    static let light = BrandPalette(
        bgInner: Tone(0xF9F5EB),
        bgMid: Tone(0xF7F2E6),
        bgOuter: Tone(0xF5EFE1),
        cardSurface: Tone(0xFBF7EE),
        raisedSurface: Tone(0xFFFCF5),
        panelSurfaceTop: Tone(0xFFFCF5),
        panelSurfaceBottom: Tone(0xFBF7EE),
        floorCenter: Tone(0xF7F2E6),
        floorEdge: Tone(0xF1EADB),
        typeHi: Tone(0x1A1612),
        typeBody: Tone(0x2B251E),
        typeSecondary: Tone(0x6B6257),
        typeTertiary: Tone(0x857C71),
        invertedType: Tone(0xFBF7EE),
        typeGradientTop: Tone(0x1A1612),
        typeGradientMid: Tone(0x2B251E),
        typeGradientBottom: Tone(0x4A423A),
        chromeStop0: Tone(0x5C6368),
        chromeStop1: Tone(0x40474C),
        chromeStop2: Tone(0x22272B),
        chromeStop3: Tone(0x40474C),
        chromeStop4: Tone(0x5C6368),
        accentChrome: Tone(0x4B5359),
        accentWarm: Tone(0x6E6252),
        coolChrome: Tone(0x3F5566),
        chromeHalo: Tone(0x3B2F1F, alpha: 0.10),
        warning: Tone(0x8F6300),
        danger: Tone(0xB4432E),
        success: Tone(0x2A7247),
        separator: Tone(0x2B251E, alpha: 0.10),
        separatorStrong: Tone(0x2B251E, alpha: 0.18),
        wash: Tone(0x2B251E),
        shade: Tone(0x3B2F1F, alpha: 0.36),
        ink: Tone(0x1F1A14),
        onInk: Tone(0xFBF7EE),
        onAccent: Tone(0xFFFFFF),
        hudFill: Tone(0xFBF7EE, alpha: 0.92),
        shadowLow: Tone(0x3B2F1F, alpha: 0.11),
        shadowMid: Tone(0x3B2F1F, alpha: 0.14),
        shadowHi: Tone(0x3B2F1F, alpha: 0.18)
    )

    /// The palette for an AppKit appearance. `bestMatch` folds the
    /// high-contrast variants onto their base look.
    static func palette(for appearance: NSAppearance) -> BrandPalette {
        appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? .dark : .light
    }

    /// An NSColor that resolves the tone from the drawing appearance.
    /// Safe to hand to AppKit views and attributed strings: they re-read
    /// it whenever their effective appearance changes.
    static func nsColor(_ tone: KeyPath<BrandPalette, Tone>) -> NSColor {
        NSColor(name: nil) { appearance in
            palette(for: appearance)[keyPath: tone].nsColor
        }
    }

    /// A SwiftUI Color that resolves the tone from the environment's
    /// color scheme (SwiftUI matches the NSColor provider to the scheme
    /// set by `preferredColorScheme`). `.opacity(_:)` multiplies the
    /// tone's own alpha, so call sites keep their literal opacities.
    static func color(_ tone: KeyPath<BrandPalette, Tone>) -> Color {
        Color(nsColor: nsColor(tone))
    }
}

// MARK: - Brand
//
// MTPLX "Jet Chrome": single source of truth for every token the app
// reads. Identity is jet black + off-white + sparing polished-steel
// chrome, with a cream counterpart selected by the user's appearance
// preference (see `ThemeStore.appearance`). Every token here is
// appearance-aware; the per-look values live in `BrandPalette`. The
// chrome accent is a desaturated steel solid (`accentChrome`) plus a
// 5-stop polished-steel `chromeAccent` gradient for the wordmark, hero
// TPS, and ALL-TIME MAX hero badge. The old `accentBlue` is deprecated
// and aliased to `accentChrome` so any missed call site still renders
// as chrome instead of regressing to the old "AI blue."

enum Brand {
    // MARK: - Surfaces (true neutral on jet; warm paper on cream)

    /// Innermost panel / card surface.
    static let bgInner = BrandPalette.color(\.bgInner)

    /// Middle background: the layer most views read against.
    static let bgMid = BrandPalette.color(\.bgMid)

    /// Outermost floor color (window background).
    static let bgOuter = BrandPalette.color(\.bgOuter)

    /// Card / raised surface: the panel a list-row or content card sits
    /// on.
    static let cardSurface = BrandPalette.color(\.cardSurface)

    /// Slightly elevated surface (chips, pucks, inner controls).
    static let raisedSurface = BrandPalette.color(\.raisedSurface)

    /// Top stop of the canonical chrome panel gradient. Used by
    /// `PanelChrome` and any per-surface panel that wants the same
    /// vertical fade.
    static let panelSurfaceTop = BrandPalette.color(\.panelSurfaceTop)

    /// Bottom stop of the canonical chrome panel gradient. Matches
    /// `cardSurface` so the panel grounds into the surrounding layout.
    static let panelSurfaceBottom = BrandPalette.color(\.panelSurfaceBottom)

    // MARK: - Type tokens (off-white on jet; warm ink on cream)

    /// Headline tier. Wordmark fallback, hero TPS digits, gauge headline.
    static let typeHi = BrandPalette.color(\.typeHi)

    /// Body tier: primary running text.
    static let typeBody = BrandPalette.color(\.typeBody)

    /// Secondary type: labels, captions, supporting metadata.
    static let typeSecondary = BrandPalette.color(\.typeSecondary)

    /// Tertiary type: the quietest metadata tier.
    static let typeTertiary = BrandPalette.color(\.typeTertiary)

    /// Type painted on a `typeHi` / `typeBody` filled pill (the inverted
    /// CTA). Black on the off-white pill in dark, paper on the ink pill
    /// in light.
    static let invertedType = BrandPalette.color(\.invertedType)

    /// Backwards-compat aliases for the old `textHighlight`/`accent`
    /// references. Resolved to the same type tiers so existing call
    /// sites keep compiling while the rest of the app moves to explicit
    /// `typeHi` / `typeBody` references.
    static let textHighlight = typeBody
    static let accent = typeHi

    // MARK: - Wordmark gradient (kept for the < 14pt fallback)

    /// Three-stop type gradient used only by the small-size text
    /// wordmark fallback in `WordmarkView` and the chrome CTA fills.
    /// The real wordmark ships as a PNG asset for any height >= 14pt.
    static let typeGradientStops: [Gradient.Stop] = [
        .init(color: BrandPalette.color(\.typeGradientTop), location: 0.0),
        .init(color: BrandPalette.color(\.typeGradientMid), location: 0.55),
        .init(color: BrandPalette.color(\.typeGradientBottom), location: 1.0),
    ]

    static let typeGradient = LinearGradient(
        stops: typeGradientStops,
        startPoint: .top,
        endPoint: .bottom
    )

    // MARK: - Chrome accents (the only color tone allowed)

    /// Polished-steel 5-stop sheen used for `MTPLXChromeText`, the gauge
    /// max tick, hero TPS, ALL-TIME MAX badge, and panel highlights.
    /// Bright top / dim middle / bright bottom so the gradient reads as
    /// light catching a curved bevel. On cream the same shape runs in
    /// graphite: lighter steel at the edges, near-black in the middle.
    static let chromeAccentStops: [Gradient.Stop] = [
        .init(color: BrandPalette.color(\.chromeStop0), location: 0.00),
        .init(color: BrandPalette.color(\.chromeStop1), location: 0.25),
        .init(color: BrandPalette.color(\.chromeStop2), location: 0.55),
        .init(color: BrandPalette.color(\.chromeStop3), location: 0.80),
        .init(color: BrandPalette.color(\.chromeStop4), location: 1.00),
    ]

    static let chromeAccent = LinearGradient(
        stops: chromeAccentStops,
        startPoint: .top,
        endPoint: .bottom
    )

    /// Desaturated steel solid. Single replacement for every
    /// `accentBlue` reference. Reads as polished metal, never as blue.
    static let accentChrome = BrandPalette.color(\.accentChrome)

    /// Warm-steel sister tone. Used only to distinguish prefill from
    /// decode on the gauge and to mark warm semantic states.
    static let accentWarm = BrandPalette.color(\.accentWarm)

    /// Deprecated. Repointed to `accentChrome` so any missed call site
    /// still renders as polished chrome instead of regressing to the old
    /// "AI blue." Migrate references to `Brand.accentChrome` (solid) or
    /// `Brand.chromeAccent` (gradient).
    @available(*, deprecated, message: "Use Brand.accentChrome (solid) or Brand.chromeAccent (gradient).")
    static let accentBlue: Color = BrandPalette.color(\.accentChrome)

    /// Cool-chrome tint reused as the gauge border when fans are pinned
    /// to max. Slightly cooler than `accentChrome` so it reads as
    /// "state," not "action."
    static let coolChrome = BrandPalette.color(\.coolChrome)

    /// Ambient halo behind chrome text and the hero TPS digits: a faint
    /// white glow on jet, a soft warm shadow on cream.
    static let chromeHalo = BrandPalette.color(\.chromeHalo)

    // MARK: - Status colors

    static let warning = BrandPalette.color(\.warning)
    static let danger = BrandPalette.color(\.danger)
    static let success = BrandPalette.color(\.success)

    // MARK: - Hairlines (tokenized, no more 1.4 / 1.5 drift)

    static let hairline: CGFloat = 0.5
    static let hairlineStrong: CGFloat = 0.75
    static let hairlineHeavy: CGFloat = 1.0

    /// Hairline separator at 6% white on jet, 10% warm ink on cream.
    /// Cards, dividers, tab-bar lines.
    static let separator = BrandPalette.color(\.separator)

    /// Slightly stronger hairline for active/selected boundaries.
    static let separatorStrong = BrandPalette.color(\.separatorStrong)

    // MARK: - Washes, shades, inks
    //
    // The dark look built every chip, hover state, and bevel edge from
    // white at a low opacity, and every shadow from black. These tokens
    // keep those call-site opacities intact and swap the base tone per
    // appearance, so `Brand.wash.opacity(0.04)` is a faint lift on jet
    // and a faint warm wash on cream.

    /// Low-opacity wash for chips, hover states, and bevel edges. Use
    /// with `.opacity(_:)`. White on jet, warm ink on cream.
    static let wash = BrandPalette.color(\.wash)

    /// Drop-shadow base. Use with `.opacity(_:)`. Black on jet; a warm
    /// umber, pre-dimmed, on cream so the same opacity reads as a soft
    /// paper shadow.
    static let shade = BrandPalette.color(\.shade)

    /// Solid ink that stays dark in both looks. Use for glyph plates and
    /// discs that sit on colored badges, with `.opacity(_:)`.
    static let ink = BrandPalette.color(\.ink)

    /// Type and glyphs painted on `ink`.
    static let onInk = BrandPalette.color(\.onInk)

    /// Type and glyphs painted on an `accentChrome` or `danger` fill.
    static let onAccent = BrandPalette.color(\.onAccent)

    /// Backdrop of the floating render HUD.
    static let hudFill = BrandPalette.color(\.hudFill)

    /// Square diameter of every chrome-strip action button: the
    /// LaunchButton play/stop, the inference-params slider button, and
    /// the refresh button all share this so they read as a uniform row
    /// of circular controls.
    static let controlSize: CGFloat = 32

    // MARK: - Elevation tokens
    //
    // Real shadows. Black at meaningful opacities on jet so cards,
    // panels, and overlays feel raised; a warm umber at lower opacities
    // on cream so they cast soft paper shadows instead of grey smudges.
    // Legacy `Depth.ambient` and `Depth.near` aliases below point at
    // `Elevation.low` so existing call sites pick up real elevation
    // without a rewrite.

    enum Elevation {
        /// Tiles, chips, low-profile controls.
        static let low = (color: BrandPalette.color(\.shadowLow), radius: 6.0, x: 0.0, y: 2.0)

        /// Cards, secondary surfaces.
        static let mid = (color: BrandPalette.color(\.shadowMid), radius: 12.0, x: 0.0, y: 6.0)

        /// Panels, overlays, sheets.
        static let hi = (color: BrandPalette.color(\.shadowHi), radius: 24.0, x: 0.0, y: 12.0)
    }

    // MARK: - Spacing tokens (4pt grid)

    enum Spacing {
        static let s1: CGFloat = 4
        static let s2: CGFloat = 8
        static let s3: CGFloat = 12
        static let s4: CGFloat = 16
        static let s5: CGFloat = 20
        static let s6: CGFloat = 24
        static let s7: CGFloat = 32
    }

    // MARK: - Corner radii (concentric)

    enum Radii {
        static let s: CGFloat = 8
        static let m: CGFloat = 12
        static let l: CGFloat = 14
        static let panel: CGFloat = 18
    }

    // MARK: - Legacy Depth aliases (now point at real elevation)
    //
    // Preserved so the existing `.shadow(color: Brand.Depth.ambient.color, ...)`
    // call sites (`BottomTabBar`, `MenuBarContent`, `TileRow`,
    // `WelcomeScreen` (dead), `Primitives`, `BottomBar`) pick up a real
    // shadow color for free until they migrate to `Brand.Elevation.*`.

    enum Depth {
        static let near = Elevation.low
        static let ambient = Elevation.low
    }

    // MARK: - Backwards-compat aliases (V1 chrome)
    //
    // These exist so the wholesale V1 view files keep compiling while we
    // simplify them. Kept untouched until each call site is migrated.

    static let chromeStops = typeGradientStops
    static let chromeFill = typeGradient
    static let warmChromeStops = typeGradientStops
    static let warmChromeFill = typeGradient
    static let shineStops: [Gradient.Stop] = [
        .init(color: wash.opacity(0.0), location: 0.0),
        .init(color: wash.opacity(0.0), location: 1.0),
    ]
    static let shineGradient = LinearGradient(
        stops: shineStops,
        startPoint: .top,
        endPoint: .bottom
    )

    /// Extrusion is dead. Empty array means callers' ForEach over
    /// `extrusionLayers` renders zero shadow layers, so the wordmark
    /// becomes a single flat off-white text.
    struct ExtrusionLayer {
        let offset: CGFloat
        let fill: Color
    }
    static let extrusionLayers: [ExtrusionLayer] = []

    /// The window floor. On jet both stops are `bgOuter`, so this renders
    /// the same flat piano black it always has (the radial pulled focus
    /// from the type and read "showroom carpet" rather than "tool"). On
    /// cream the two stops differ slightly, giving a subtle warm vignette
    /// instead of a flat white blast.
    static let pianoRadial = RadialGradient(
        gradient: Gradient(colors: [
            BrandPalette.color(\.floorCenter),
            BrandPalette.color(\.floorEdge),
        ]),
        center: .center,
        startRadius: 0,
        endRadius: 900
    )
}

// MARK: - BrandFont

/// Minimal Apple-ish typography. SF Pro Rounded for the wordmark + hero
/// numbers, SF Pro for body, SF Mono only for actual data. No Inter
/// dependency, system fonts only.
enum BrandFont {
    /// Wordmark / hero number. Lighter weight than V0 (was .black) for
    /// a less aggressive read at large sizes.
    static func wordmark(size: CGFloat) -> Font {
        Font.system(size: size, weight: .heavy, design: .rounded)
    }

    /// Tracking value used by callers that previously did manual
    /// kerning. The new typography uses default tracking, so zero.
    static func wordmarkTracking(size: CGFloat) -> CGFloat { 0 }

    /// Subtitle line under the wordmark. SF Pro Regular at 12pt.
    static func subtitle(size: CGFloat = 12) -> Font {
        Font.system(size: size, weight: .regular, design: .default)
    }
}
