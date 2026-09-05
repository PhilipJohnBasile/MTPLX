import SwiftUI

// MARK: - OverlayChoreography
//
// Shared entry/exit animations and row-hover treatment for the
// chrome-strip popovers (LaunchOverlay, InferenceParamsOverlay,
// ModelPickerOverlay). Each popover owns its own `borderProgress`,
// `headerVisible`, and `rowsVisibleCount` @State, but the actual spring
// and stagger choreography is centralised here so all three overlays
// look and feel identical. All timings come from `Motion`.

enum OverlayChoreography {
    /// Reset all three state values to zero, then animate the border,
    /// header, and rows in with the standard stagger. Pass the bindings
    /// to the three overlay @State fields.
    static func runEnter(
        motionEnabled: Bool,
        rowCount: Int,
        borderProgress: Binding<CGFloat>,
        headerVisible: Binding<Bool>,
        rowsVisibleCount: Binding<Int>
    ) {
        borderProgress.wrappedValue = 0
        headerVisible.wrappedValue = false
        rowsVisibleCount.wrappedValue = 0

        guard motionEnabled else {
            borderProgress.wrappedValue = 1
            headerVisible.wrappedValue = true
            rowsVisibleCount.wrappedValue = rowCount
            return
        }

        withAnimation(Motion.overlayBorder) {
            borderProgress.wrappedValue = 1
        }

        // Header + per-row staggered reveals via modern Swift Concurrency
        // (was DispatchQueue.main.asyncAfter chains per swiftui-pro
        // swift.md "Never use Grand Central Dispatch").
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(Motion.overlayHeaderDelay))
            withAnimation(Motion.overlayHeaderSpring) {
                headerVisible.wrappedValue = true
            }
        }

        for idx in 0..<rowCount {
            let delay = Motion.overlayRowBaseDelay + Double(idx) * Motion.overlayRowStaggerStep
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(delay))
                withAnimation(Motion.overlayRowSpring) {
                    rowsVisibleCount.wrappedValue = idx + 1
                }
            }
        }
    }

    /// Animate the overlay out. `additionalReset` is a hook for
    /// surface-specific cleanup (e.g. collapsing an inline form) that
    /// should ride the same animation transaction.
    static func runExit(
        motionEnabled: Bool,
        borderProgress: Binding<CGFloat>,
        headerVisible: Binding<Bool>,
        rowsVisibleCount: Binding<Int>,
        additionalReset: (() -> Void)? = nil
    ) {
        let animation: Animation? = motionEnabled ? Motion.overlayExit : nil
        withAnimation(animation) {
            borderProgress.wrappedValue = 0
            headerVisible.wrappedValue = false
            rowsVisibleCount.wrappedValue = 0
            additionalReset?()
        }
    }
}

// MARK: - OverlayRowHover
//
// Pop on hover: subtle white-on-white card behind the row that lifts
// via shadow, plus a 1.02 magnify so the lift reads as a pop rather
// than a colour change. Lifted from the original `LaunchRow`
// implementation, parameterised on the motion flag, exposed so the
// inference and model overlays can adopt identical interaction feel.

struct OverlayRowHover: ViewModifier {
    let motionEnabled: Bool

    @State private var hovering: Bool = false

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: Motion.rowHoverCornerRadius, style: .continuous)
                    .fill(hovering ? Motion.rowHoverFill : Color.clear)
                    .padding(.horizontal, Motion.rowHoverInset)
                    .shadow(
                        color: Brand.shade.opacity(hovering ? Motion.rowHoverShadowOpacity : 0),
                        radius: hovering ? Motion.rowHoverShadowRadius : 0,
                        x: 0,
                        y: hovering ? Motion.rowHoverShadowYOffset : 0
                    )
            )
            .scaleEffect(hovering ? Motion.rowHoverScale : 1.0)
            .animation(motionEnabled ? Motion.rowHoverSpring : nil, value: hovering)
            .onHover { isHovering in
                if isHovering { Haptics.tick(.levelChange) }
                hovering = isHovering
            }
    }
}

extension View {
    /// Apply the chrome-strip popover row hover treatment (scale 1.02 +
    /// soft white fill + drop shadow). Used by LaunchRow, InferenceRow,
    /// and modelRow.
    func overlayRowHover(motionEnabled: Bool) -> some View {
        modifier(OverlayRowHover(motionEnabled: motionEnabled))
    }
}
