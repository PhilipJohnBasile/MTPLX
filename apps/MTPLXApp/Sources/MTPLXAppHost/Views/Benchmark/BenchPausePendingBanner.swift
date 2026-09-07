import SwiftUI
import MTPLXAppCore

// MARK: - BenchPausePendingBanner
//
// In-overlay banner shown while a hard pause is being sent to the
// active decode. Warning-amber surface; "Cancel pause" inline button
// flips the queued pause back to resume.

struct BenchPausePendingBanner: View {
    let pendingProblemIdx: Int
    let onCancelPause: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "pause.circle.fill")
                .foregroundStyle(Brand.warning)
            Text(tr("Stopping problem %lld...", pendingProblemIdx))
                .font(.caption)
                .foregroundStyle(Brand.typeBody)
            Spacer()
            Button(tr("Cancel pause"), action: onCancelPause)
                .buttonStyle(.plain)
                .font(.caption.weight(.semibold))
                .foregroundStyle(Brand.warning)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background {
            RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                .fill(Brand.warning.opacity(0.10))
                .overlay {
                    RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                        .strokeBorder(Brand.warning.opacity(0.35), lineWidth: Brand.hairline)
                }
        }
    }
}
