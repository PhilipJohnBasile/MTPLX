import SwiftUI
import MTPLXAppCore

// MARK: - LanguagePromptSheet
//
// The one-time language prompt for installs that finished onboarding
// before the Language step existed (everything shipped before 2.11).
// MTPLXApp raises it on the first launch after the update when
// `MTPLXAppConfiguration.shouldOfferLanguagePrompt` is true; ContentView
// stamps `languagePromptCompletedAt` on dismiss, so it appears exactly
// once however it is closed. Fresh installs pick their language in
// onboarding and never see this sheet.
//
// The picker applies each pick live, so this very sheet re-renders in
// the language just chosen — that is the point: people learn the app
// speaks theirs, and the subtitle tells them where to change it later.

struct LanguagePromptSheet: View {
    @EnvironmentObject private var languageStore: LanguageStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            LanguagePickerList()
                .frame(height: 320)
                .padding(.top, 20)
            footer
                .padding(.top, 20)
        }
        .padding(28)
        .frame(width: 440)
        .background(Brand.pianoRadial.ignoresSafeArea())
        // A sheet is its own window on macOS: it applies the appearance
        // preference itself and follows the picked language's locale and
        // layout direction the way the main window does.
        .appliesAppearance()
        .environment(\.locale, languageStore.language.locale)
        .environment(\.layoutDirection, languageStore.language.isRightToLeft ? .rightToLeft : .leftToRight)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(tr("Choose your language"))
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(tr("Choose your language"))
                .font(.system(.title2, design: .rounded).weight(.semibold))
                .foregroundStyle(Brand.typeHi)
            Text(tr("MTPLX speaks %lld languages. Your pick applies right away, and you can change it anytime in Settings.", AppLanguage.allCases.count))
                .font(.system(size: 13))
                .foregroundStyle(Brand.typeSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var footer: some View {
        HStack {
            Spacer(minLength: 0)
            Button(tr("Continue")) { dismiss() }
                .buttonStyle(MTPLXPillButton())
                .accessibilityHint(tr("Keeps %@ and closes this prompt.", languageStore.language.nativeName))
        }
    }
}
