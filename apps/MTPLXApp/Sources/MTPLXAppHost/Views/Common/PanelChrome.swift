import SwiftUI

// MARK: - PanelChrome
//
// Canonical Jet Chrome panel. The chrome panel gradient + 3-stop chrome
// stroke + real elevation that was previously hand-rolled at four
// different sites (BenchmarkOverlay panel chrome, BenchLiveCard chrome,
// BenchSummaryCard chrome, ForgeCreateView stage shell) — pulled into a
// single primitive so every panel reads with the same surface tone,
// the same hairline weight, and the same drop shadow.
//
// Used as a background under any content that wants the canonical
// panel treatment:
//
//   content
//       .padding(20)
//       .background { PanelChrome(cornerRadius: Brand.Radii.panel) }
//
// The corner radius defaults to `Brand.Radii.panel` (18) so calls that
// want the canonical size can omit the parameter. Smaller cards pass
// `Brand.Radii.l` (14) or `Brand.Radii.m` (12).

struct PanelChrome: View {
    var cornerRadius: CGFloat
    var elevation: (color: Color, radius: Double, x: Double, y: Double)

    init(
        cornerRadius: CGFloat = Brand.Radii.panel,
        elevation: (color: Color, radius: Double, x: Double, y: Double) = Brand.Elevation.hi
    ) {
        self.cornerRadius = cornerRadius
        self.elevation = elevation
    }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        shape
            .fill(
                LinearGradient(
                    colors: [Brand.panelSurfaceTop, Brand.panelSurfaceBottom],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .overlay {
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
            }
            .shadow(
                color: elevation.color,
                radius: elevation.radius,
                x: elevation.x,
                y: elevation.y
            )
    }
}
