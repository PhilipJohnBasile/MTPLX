import SwiftUI
import MTPLXAppCore

// MARK: - LanguageStep
//
// First onboarding step: pick the app language before anything else is
// shown, so the rest of the flow renders in it. The list applies the
// choice live (the step observes LanguageStore), and Settings keeps the
// same picker for later changes.

struct LanguageStep: View {
    @ObservedObject var orchestrator: OnboardingOrchestrator
    @EnvironmentObject private var languageStore: LanguageStore

    var body: some View {
        OnboardingStepContainer(
            title: tr("Choose your language"),
            subtitle: tr("MTPLX will use this language everywhere. You can change it later in Settings."),
            stepIndex: OnboardingStep.language.index,
            stepCount: OnboardingStep.allCases.count,
            primary: {
                OnboardingPrimaryButton(tr("Continue")) { orchestrator.goNext() }
            },
            content: {
                LanguagePickerList()
                    .frame(height: 320)
            }
        )
    }
}
