import SwiftUI
import MTPLXAppCore

// MARK: - ChatEmptyView
//
// Welcome state for a brand-new conversation. Centred wordmark at 30%
// opacity above a soft subtitle, matching Aphanes' empty composition
// but using MTPLX Brand.

struct ChatEmptyView: View {
    var body: some View {
        VStack(spacing: 12) {
            WordmarkView(height: 48)
                .opacity(0.30)
            Text(tr("Ask anything to get started."))
                .font(BrandFont.subtitle())
                .foregroundStyle(Brand.typeSecondary)
            Text(tr("Attach files with the paperclip. Tap the globe to search the web."))
                .font(.system(size: 11))
                .foregroundStyle(Brand.typeTertiary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 360)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.vertical, 40)
    }
}
