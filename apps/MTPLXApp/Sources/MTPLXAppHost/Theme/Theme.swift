import SwiftUI

// MARK: - AppAppearance

/// The user's appearance preference. `dark` is the brand's first-run
/// look (Jet Chrome), `light` is the cream counterpart, and `system`
/// follows the macOS Appearance setting. Persisted by `ThemeStore`.
public enum AppAppearance: String, CaseIterable, Identifiable, Sendable {
    case system
    case dark
    case light

    public var id: String { rawValue }

    /// The scheme handed to `preferredColorScheme`. `nil` lets the
    /// window follow macOS.
    public var colorScheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .dark: return .dark
        case .light: return .light
        }
    }

    /// Segmented-control label.
    public var title: String {
        switch self {
        case .system: return "System"
        case .dark: return "Dark"
        case .light: return "Light"
        }
    }
}

// MARK: - ThemeStore
//
// MTPLX V1 is one curated brand identity (piano black + chrome silver)
// with a cream counterpart. The old multi-theme picker
// (system/hippo/river/mono) is gone; what the store holds are the
// preferences that genuinely belong to the user: which of the two looks
// to render (or whether to follow macOS), whether they want subtle sound
// cues, and whether they want motion reduced for accessibility.

@MainActor
public final class ThemeStore: ObservableObject {
    /// UserDefaults key of the appearance preference.
    public static let appearanceKey = "mtplx.app.appearance"

    @AppStorage("mtplx.app.soundEnabled") public var soundEnabled: Bool = false
    @AppStorage("mtplx.app.reduceMotion") public var reduceMotionPreference: Bool = false

    /// Appearance preference. Defaults to `dark` so a first launch renders
    /// exactly the Jet Chrome identity it always has.
    @AppStorage("mtplx.app.appearance") public var appearance: AppAppearance = .dark

    public init() {}
}

// MARK: - Modifiers

/// Applies the store's appearance preference to the enclosing window
/// via `preferredColorScheme`. Sheets and the menu bar popover are
/// separate windows on macOS, so each root applies this itself.
public struct AppliesAppearance: ViewModifier {
    @EnvironmentObject private var themeStore: ThemeStore

    public init() {}

    public func body(content: Content) -> some View {
        content.preferredColorScheme(themeStore.appearance.colorScheme)
    }
}

/// Pins the window to the MTPLX brand identity:
/// - `preferredColorScheme` from the store, so the tokens resolve to the
///   look the user chose regardless of the macOS Appearance setting.
/// - `.tint(Brand.accent)` so all standard controls pick up the brand tone.
/// - `.background(Brand.bgOuter)` as the floor color (the piano radial
///   sits on top of this in `ContentView`).
public struct AppliesBrand: ViewModifier {
    public init() {}

    public func body(content: Content) -> some View {
        content
            .modifier(AppliesAppearance())
            .tint(Brand.accent)
            .background(Brand.bgOuter)
    }
}

public extension View {
    /// Apply at the root of the view hierarchy to lock the MTPLX brand.
    func appliesBrand() -> some View {
        modifier(AppliesBrand())
    }

    /// Apply at the root of every separately-windowed surface (sheets,
    /// the menu bar popover) so it follows the appearance preference.
    func appliesAppearance() -> some View {
        modifier(AppliesAppearance())
    }
}

// MARK: - Semantic color helpers
//
// Kept as `mtplx*` to preserve call-site stability across the codebase.
// They now resolve to `Brand` tokens so any future tweak ripples
// everywhere.

extension Color {
    /// Subtle separator on card boundaries and tab-bar hairlines.
    static var mtplxSeparator: Color { Brand.separator }

    /// Warning amber used by ThermalRuleBanner + NewMaxToast.
    static var mtplxWarning: Color { Brand.warning }

    /// Danger red used by ConnectionIssueBanner + degraded states.
    static var mtplxDanger: Color { Brand.danger }

    /// Calm success tint used by fan-verified / cache-hit indicators.
    static var mtplxSuccess: Color { Brand.success }

    /// Default accent, polished chrome. Use this anywhere the old code
    /// reached for `Color.accentColor`. Resolves to the desaturated
    /// steel solid so toolbar tints, focus rings, and system controls
    /// all pick up the brand identity.
    static var mtplxAccent: Color { Brand.accentChrome }
}

// MARK: - Motion

/// Wraps `withAnimation` so Performance Lock and the user's Reduce Motion
/// preference can short-circuit animation without touching every call site.
@inlinable public func animateValue<V>(
    _ animation: Animation,
    motionEnabled: Bool,
    _ body: () -> V
) -> V {
    if motionEnabled {
        return withAnimation(animation, body)
    }
    return body()
}
