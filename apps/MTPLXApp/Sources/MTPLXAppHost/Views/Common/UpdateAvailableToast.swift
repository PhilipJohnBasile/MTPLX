import SwiftUI
import MTPLXAppCore

// MARK: - UpdateAvailableToast
//
// Bottom-right gentle reminder shown when a scheduled Sparkle check
// finds a new version. Styled like a dashboard tile: monospaced
// heavy tracked header, panel-gradient surface, hairline border.
// "Install Now" resumes the Sparkle session in focus; "Later" hides
// the toast until the next scheduled check finds the update again.

struct UpdateAvailableToast: View {
    let update: MTPLXAppUpdater.PendingUpdate
    let onInstall: () -> Void
    let onLater: () -> Void

    var body: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(tr("UPDATE AVAILABLE"))
                    .font(.system(size: 11, weight: .heavy, design: .monospaced))
                    .tracking(3)
                    .foregroundStyle(Brand.typeSecondary)
                Text(tr("MTPLX %@ (%@)", update.version, update.build))
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
            }

            Spacer(minLength: 18)

            Button(action: onLater) {
                Text(tr("Later"))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Brand.typeSecondary)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 6)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(tr("Remind me later"))

            Button(action: onInstall) {
                Text(tr("Install Now"))
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Brand.invertedType)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 6)
                    .background {
                        Capsule(style: .continuous)
                            .fill(Brand.typeHi)
                    }
                    .contentShape(Capsule(style: .continuous))
            }
            .buttonStyle(.plain)
            .accessibilityLabel(tr("Install update now"))
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
        .background {
            RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [Brand.panelSurfaceTop, Brand.panelSurfaceBottom],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .overlay {
                    RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous)
                        .strokeBorder(Brand.wash.opacity(0.12), lineWidth: Brand.hairline)
                }
                .shadow(color: Brand.shade.opacity(0.55), radius: 20, y: 8)
        }
        .frame(maxWidth: 440)
    }
}
