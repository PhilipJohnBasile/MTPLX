import SwiftUI
import MTPLXAppCore

// MARK: - RegisteredStage
//
// Celebration card shown after Forge completes. Surfaces the
// verification headline + best-case TPS multiplier (via the same
// AcceptanceReveal primitives the Verify stage uses, but as a
// summary block rather than a per-depth grid). Three peer CTAs
// reflect the three things a user does next:
//
//   • Use it now      → swap the daemon to the new local path and
//                        flip to Live to watch it start
//   • Publish to HF   → opens the upload form; upload starts after
//                        the user supplies repo/token
//   • Build another   → resets the wizard back to Source

struct RegisteredStage: View {
    @EnvironmentObject private var orchestrator: ForgeOrchestrator
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var router: AppRouter

    @State private var isLaunching: Bool = false

    var body: some View {
        ForgeStageShell(
            title: title,
            subtitle: subtitle,
            step: .registered,
            symbol: "checkmark.seal.fill",
            symbolTint: Brand.success
        ) {
            VStack(alignment: .leading, spacing: 18) {
                celebrationCard
                ctaRow
                pathRow
                Spacer(minLength: 0)
            }
            .onAppear { registerInCustomModels() }
        } footer: {
            EmptyView()
        }
    }

    // MARK: - Picker registration
    //
    // On reach (not on Use-it-now), the forged model is persisted to
    // `AppConfiguration.customModels` so the chrome-strip model
    // picker surfaces it for switching from anywhere in the app —
    // not only from the Forge tab. Idempotent: `rememberForgedModel`
    // dedups by id + local path.

    private func registerInCustomModels() {
        guard orchestrator.state.hasSpeedWinningVerification else { return }
        guard let path = orchestrator.completedLocalPath else { return }
        let brandedName = orchestrator.state.brand.brandedName
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !brandedName.isEmpty else { return }
        let sizeBytes = orchestrator.state.sourceProbe?.estimatedSizeBytes ?? 0
        let peakGiB = orchestrator.state.sourceProbe?.estimatedPeakGiB ?? 0
        var config = backend.configuration
        config.rememberForgedModel(
            brandedName: brandedName,
            localPath: path,
            sizeBytes: sizeBytes,
            peakMemoryGiB: peakGiB
        )
        try? backend.saveSettings(config)
    }

    private var title: String {
        if let verification = orchestrator.state.verification, orchestrator.state.hasSpeedWinningVerification {
            return tr("Forged — %.2f× faster than baseline", verification.multiplierVsAr)
        }
        return tr("Forged")
    }

    private var subtitle: String? {
        guard let path = orchestrator.completedLocalPath else {
            return tr("Your model is ready and stamped with MTPLX.")
        }
        return tr("Saved to %@.", path.replacingOccurrences(of: NSHomeDirectory(), with: "~"))
    }

    // MARK: - Celebration card

    @ViewBuilder
    private var celebrationCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "checkmark.seal.fill")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(Brand.success)
                VStack(alignment: .leading, spacing: 4) {
                    Text(orchestrator.state.brand.brandedName.isEmpty
                         ? "Your forged model"
                         : orchestrator.state.brand.brandedName)
                        .font(.system(.title3, design: .rounded).weight(.semibold))
                        .foregroundStyle(Brand.typeHi)
                    if let verification = orchestrator.state.verification {
                        Text(verificationLine(verification))
                            .font(.system(size: 12))
                            .foregroundStyle(Brand.typeSecondary)
                    }
                }
                Spacer(minLength: 0)
            }
            if let verification = orchestrator.state.verification {
                AcceptanceRevealTPSPanel(data: AcceptanceRevealData.from(verification))
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    private func verificationLine(_ v: ForgeVerification) -> String {
        let best = v.tokSByDepth[v.bestDepth] ?? 0
        return tr("%.1f tok/s · %.0f%% mean accept at D%d · verified on %@",
                      best,
                      avgAcceptance(v) * 100,
                      v.bestDepth,
                      v.verifiedOnHardware)
    }

    private func avgAcceptance(_ v: ForgeVerification) -> Double {
        let row = v.acceptanceByDepth[v.bestDepth] ?? []
        guard !row.isEmpty else { return 0 }
        return row.reduce(0, +) / Double(row.count)
    }

    // MARK: - CTA row (three peers)

    @ViewBuilder
    private var ctaRow: some View {
        HStack(spacing: 12) {
            ForgePrimaryButton("Use it now", icon: "play.fill", isEnabled: !isLaunching && orchestrator.state.hasSpeedWinningVerification) {
                useItNow()
            }
            secondaryButton(tr("Publish to HF"), icon: "arrow.up.circle.fill") {
                orchestrator.openPublishStage()
            }
            secondaryButton(tr("Build another"), icon: "plus") {
                orchestrator.resetWizard()
            }
            Spacer(minLength: 0)
        }
    }

    private func secondaryButton(_ title: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: icon)
                    .font(.system(size: 11, weight: .semibold))
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .foregroundStyle(Brand.typeBody)
            .background(
                Capsule(style: .continuous)
                    .fill(Brand.cardSurface)
                    .overlay(
                        Capsule(style: .continuous)
                            .stroke(Brand.separator, lineWidth: 0.5)
                    )
            )
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var pathRow: some View {
        if let path = orchestrator.completedLocalPath {
            HStack(spacing: 6) {
                Image(systemName: "folder")
                    .font(.system(size: 10))
                    .foregroundStyle(Brand.typeTertiary)
                Text(path)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .truncationMode(.middle)
                    .lineLimit(1)
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                } label: {
                    Image(systemName: "arrow.up.right.square")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Brand.typeSecondary)
                }
                .buttonStyle(.plain)
                .help(tr("Reveal in Finder"))
                .accessibilityLabel(tr("Reveal in Finder"))
                Spacer(minLength: 0)
            }
        }
    }

    // MARK: - "Use it now" handoff
    //
    // Persists the new model path + the picker registration in
    // AppConfiguration, applies the configuration to the backend
    // (restarting the daemon against the new model if it's already
    // running), and flips the user to the Live tab so they can watch
    // it warm up. Heavy-lifting around customModels lands in todo 14.

    private func useItNow() {
        guard orchestrator.state.hasSpeedWinningVerification else { return }
        guard let path = orchestrator.completedLocalPath else { return }
        isLaunching = true
        var config = backend.configuration
        if let verification = orchestrator.state.verification {
            config.applyForgeRuntimeDefaults(
                modelPath: path,
                verification: verification,
                sourceRepo: orchestrator.state.sourceProbe?.hfRepo
            )
        } else {
            config.model = path
        }
        try? backend.saveSettings(config)
        Task {
            try? await backend.applyConfiguration(config, restartIfRunning: true)
            await MainActor.run {
                router.select(.live)
                router.primaryMode = .dashboard
                isLaunching = false
            }
        }
    }
}
