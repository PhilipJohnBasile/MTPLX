import SwiftUI
import MTPLXAppCore

// MARK: - EmptyStateView
//
// Used by tabs when the daemon is stopped or a snapshot is
// unavailable. Branded glyph + headline (in polished chrome) + body
// copy + optional CTA. Title uses `chromeText()` so the empty-state
// hero reads as Jet Chrome rather than as flat off-white.

struct EmptyStateView: View {
    let symbol: String
    let title: String
    let message: String
    var action: (() -> Void)? = nil
    var actionLabel: String = tr("Start")

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: symbol)
                .font(.system(size: 36, weight: .light))
                .foregroundStyle(Brand.typeBody.opacity(0.45))
            Text(title)
                .font(.system(.title3, design: .rounded).weight(.bold))
                .chromeText()
            Text(message)
                .font(.callout)
                .foregroundStyle(Brand.typeBody.opacity(0.7))
                .multilineTextAlignment(.center)
                .frame(maxWidth: 360, minHeight: 38)
            if let action {
                Button(actionLabel, action: action)
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)
                    .tint(Brand.accentChrome)
                    .padding(.top, 4)
            }
        }
        .padding(.horizontal, 32)
        .padding(.vertical, 48)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
