import SwiftUI
import MTPLXAppCore

// MARK: - BenchErrorBanner
//
// In-overlay banner that surfaces an orchestrator error. Danger-red
// surface; inline Dismiss clears the banner without resetting the run
// state.

struct BenchErrorBanner: View {
    let message: String
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Brand.danger)
            Text(message)
                .font(.caption)
                .foregroundStyle(Brand.typeBody)
                .lineLimit(2)
            Spacer()
            Button(tr("Dismiss"), action: onDismiss)
                .buttonStyle(.plain)
                .font(.caption.weight(.semibold))
                .foregroundStyle(Brand.danger)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background {
            RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                .fill(Brand.danger.opacity(0.10))
                .overlay {
                    RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                        .strokeBorder(Brand.danger.opacity(0.35), lineWidth: Brand.hairline)
                }
        }
    }
}
