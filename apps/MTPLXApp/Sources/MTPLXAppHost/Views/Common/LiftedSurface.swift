import SwiftUI

// MARK: - LiftedSurface
//
// State-driven elevation primitive for the LiveTab dashboard. Renders
// as a flat `cardSurface` tile (matching `LiveTile`'s previous look)
// when `lifted == false`, and crossfades into the canonical chrome
// panel treatment (matching `PanelChrome`) when `lifted == true`. The
// drop shadow interpolates between `Brand.Elevation.low` and the
// configured `liftedElevation` over the same animation so the tile
// physically raises off the page as the chrome catches.
//
// The two visual states are stacked rather than swapped: the flat
// base is always present so the chrome layer's gradient fades down
// onto it, which avoids the "snap to a different fill" feel that an
// `if lifted { … } else { … }` branch would produce.
//
// Drives the "model just finished loading" reveal on the Live tab.
// When the daemon reaches `.running`, `TileRow`, `AcceptanceSection`,
// `DecodeChart`, and `VerifyWaterfallExpander` all flip `lifted` to
// `true` with a staggered `delay` so the dashboard reads as
// powering-on rather than snapping. Stop the daemon and the same
// stagger settles every surface back to flat.
//
// Use:
//
//   content
//       .padding(14)
//       .background {
//           LiftedSurface(
//               lifted: isRunning,
//               cornerRadius: Brand.Radii.m,
//               delay: Double(index) * Motion.surfaceLiftStaggerStep
//           )
//       }

struct LiftedSurface: View {
    var lifted: Bool
    var cornerRadius: CGFloat = Brand.Radii.l
    var animation: Animation? = Motion.surfaceLiftSpring
    var delay: TimeInterval = 0
    var idleElevation: (color: Color, radius: Double, x: Double, y: Double) = Brand.Elevation.low
    var liftedElevation: (color: Color, radius: Double, x: Double, y: Double) = Brand.Elevation.mid

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        ZStack {
            shape
                .fill(Brand.cardSurface)

            shape
                .strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                .opacity(lifted ? 0 : 1)

            shape
                .fill(
                    LinearGradient(
                        colors: [Brand.panelSurfaceTop, Brand.panelSurfaceBottom],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .opacity(lifted ? 1 : 0)

            shape
                .strokeBorder(
                    LinearGradient(
                        stops: [
                            .init(color: Brand.wash.opacity(0.16), location: 0.0),
                            .init(color: Brand.wash.opacity(0.05), location: 0.5),
                            .init(color: Brand.wash.opacity(0.03), location: 1.0),
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    ),
                    lineWidth: Brand.hairlineStrong
                )
                .opacity(lifted ? 1 : 0)
        }
        .shadow(
            color: lifted ? liftedElevation.color : idleElevation.color,
            radius: lifted ? liftedElevation.radius : idleElevation.radius,
            x: lifted ? liftedElevation.x : idleElevation.x,
            y: lifted ? liftedElevation.y : idleElevation.y
        )
        .animation(resolvedAnimation, value: lifted)
    }

    private var resolvedAnimation: Animation? {
        guard let animation else { return nil }
        return delay > 0 ? animation.delay(delay) : animation
    }
}
