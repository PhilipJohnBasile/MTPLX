import SwiftUI

// MARK: - MTPLXChromeText
//
// Paints text (or any small graphic) with the polished-steel
// `chromeAccent` 5-stop gradient plus a 1px-equivalent specular
// highlight slice at the top of the glyphs and a soft ambient
// halo. Reads as light catching curved metal — what the wordmark PNG
// shows but produced live for the < 14pt fallback, the gauge hero TPS
// number, and the ALL-TIME MAX badge.
//
// The specular highlight is the duplicate-content technique: the same
// glyphs are drawn again at 25% white, offset up by half a point, and
// blended `.screen` so the brightest top edge of each letter catches
// an extra hit of light. Cheap, sharp, scales with font size.

struct MTPLXChromeText: ViewModifier {
    func body(content: Content) -> some View {
        content
            .foregroundStyle(Brand.chromeAccent)
            .overlay {
                content
                    .foregroundStyle(Color.white.opacity(0.25))
                    .blendMode(.screen)
                    .offset(y: -0.5)
                    .allowsHitTesting(false)
            }
            .shadow(color: Brand.chromeHalo, radius: 8)
    }
}

extension View {
    /// Apply the canonical Jet Chrome polished-steel text treatment.
    /// Use on the wordmark fallback, hero TPS digits, and ALL-TIME MAX
    /// hero badge — the three documented chrome-text sites.
    func chromeText() -> some View {
        modifier(MTPLXChromeText())
    }
}
