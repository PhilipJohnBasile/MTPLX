import SwiftUI

// MARK: - HoverElevate
//
// Consolidates the "control hover lift" pattern that the inference,
// model, and launch overlays each rolled inline — an upward 1.5pt
// offset, a soft drop shadow under the control, and the
// `Motion.controlHoverSpring` so multiple overlays float their
// controls with identical timing.
//
// Use:
//
//   slider
//       .hoverElevate(motionEnabled: !reduceMotion)
//
// Pass `motionEnabled: false` to suppress the lift entirely; the
// shadow and offset both short-circuit so the affordance falls back
// to flat without any spring.

struct HoverElevate: ViewModifier {
    var motionEnabled: Bool

    @State private var hovering = false

    func body(content: Content) -> some View {
        content
            .offset(y: liftOffset)
            .shadow(
                color: Brand.shade.opacity(shadowOpacity),
                radius: Motion.controlHoverShadowRadius,
                x: 0,
                y: Motion.controlHoverShadowYOffset
            )
            .animation(motionEnabled ? Motion.controlHoverSpring : nil, value: hovering)
            .onHover { hovering = $0 }
    }

    private var liftOffset: CGFloat {
        guard motionEnabled, hovering else { return 0 }
        return Motion.controlHoverOffsetY
    }

    private var shadowOpacity: Double {
        guard motionEnabled, hovering else { return 0 }
        return Motion.controlHoverShadowOpacity
    }
}

extension View {
    /// Canonical hover-lift treatment: a 1.5pt upward float plus a soft
    /// drop shadow on the `Motion.controlHoverSpring`. Reuses the
    /// already-tuned `Motion` constants so the lift feels identical to
    /// the overlay row hovers and the slider thumb lift.
    func hoverElevate(motionEnabled: Bool = true) -> some View {
        modifier(HoverElevate(motionEnabled: motionEnabled))
    }
}
