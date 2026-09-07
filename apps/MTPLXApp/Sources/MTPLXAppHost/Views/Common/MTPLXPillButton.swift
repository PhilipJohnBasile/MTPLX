import SwiftUI

// MARK: - MTPLXPillButton
//
// Canonical Jet Chrome pill CTA. Replaces the bespoke BenchPrimaryCTA,
// the duplicated ModelDownloadPrimaryButtonStyle, and several inline
// capsule-fill buttons sprinkled around the app. Two variants:
//
//   .primary  — off-white capsule + jet-black text. The main action on
//               each surface (Run AIME 2026, Run again, Start serving).
//   .danger   — warm-red capsule + white text. Destructive actions
//               (Cancel run, Stop).
//
// Press collapses to scale 0.96 on a quiet spring; hover lights a
// chromeAccent halo so the affordance is the surface lighting rather
// than a color change. Respects Reduce Motion (no spring, no halo
// animation when reduced).

struct MTPLXPillButton: ButtonStyle {
    enum Variant {
        case primary
        case danger
    }

    var variant: Variant = .primary

    func makeBody(configuration: Configuration) -> some View {
        MTPLXPillButtonBody(configuration: configuration, variant: variant)
    }
}

private struct MTPLXPillButtonBody: View {
    let configuration: ButtonStyle.Configuration
    let variant: MTPLXPillButton.Variant

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false

    var body: some View {
        let isDanger = variant == .danger
        configuration.label
            .font(.system(size: 13, weight: .semibold, design: .rounded))
            .foregroundStyle(isDanger ? Brand.onAccent : Brand.bgOuter)
            .padding(.horizontal, Brand.Spacing.s4)
            .padding(.vertical, 9)
            .background {
                Capsule(style: .continuous)
                    .fill(isDanger ? Brand.danger : Brand.typeBody)
            }
            .overlay {
                Capsule(style: .continuous)
                    .strokeBorder(
                        Brand.wash.opacity(0.06),
                        lineWidth: Brand.hairline
                    )
            }
            .scaleEffect(configuration.isPressed ? 0.96 : 1.0)
            .shadow(
                color: Brand.accentChrome.opacity(haloOpacity),
                radius: 12
            )
            .animation(
                reduceMotion ? nil : .spring(response: 0.18, dampingFraction: 0.72),
                value: configuration.isPressed
            )
            .animation(
                reduceMotion ? nil : .easeOut(duration: 0.18),
                value: hovering
            )
            .onHover { hovering = $0 }
            .contentShape(Capsule(style: .continuous))
    }

    private var haloOpacity: Double {
        guard hovering && !configuration.isPressed else { return 0 }
        return 0.10
    }
}

extension ButtonStyle where Self == MTPLXPillButton {
    static var mtplxPrimary: MTPLXPillButton { MTPLXPillButton(variant: .primary) }
    static var mtplxDanger: MTPLXPillButton { MTPLXPillButton(variant: .danger) }
}
