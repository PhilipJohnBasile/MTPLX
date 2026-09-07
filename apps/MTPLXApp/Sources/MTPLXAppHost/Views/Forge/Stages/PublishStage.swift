import SwiftUI
import MTPLXAppCore

// MARK: - PublishStage
//
// HF upload flow. Three logical sections in one column:
//
//   1. Identity — token paste (or "use saved" pill) + handle field,
//      both persisted in Keychain / AppConfiguration respectively
//   2. Destination — repo name + visibility radio + license dropdown
//   3. README preview — auto-generated from runtime metadata + a
//      collapsible "edit" mode for users who want to tweak
//
// Primary CTA flips between "Publish" / "Cancel upload" depending on
// orchestrator.isPublishing. Failure banner sits inline.

struct PublishStage: View {
    @EnvironmentObject private var orchestrator: ForgeOrchestrator
    @EnvironmentObject private var backend: MTPLXBackendStore

    @State private var token: String = ""
    @State private var handle: String = ""
    @State private var repoName: String = ""
    @State private var visibility: ForgePublishOptions.Visibility = .publicRepo
    @State private var license: String = "apache-2.0"
    @State private var readme: String = ""
    @State private var readmeEditing: Bool = false
    @State private var hasSavedToken: Bool = false

    private let licenses: [(spdx: String, label: String)] = [
        ("apache-2.0", "Apache 2.0"),
        ("mit", "MIT"),
        ("bsd-3-clause", "BSD-3-Clause"),
        ("llama3", "Llama 3 Community"),
        ("qwen", "Qwen License"),
        ("other", "Other (specify in README)")
    ]

    var body: some View {
        ForgeStageShell(
            title: tr("Publish to Hugging Face"),
            subtitle: tr("Share your build with the world. Your token stays in the macOS Keychain — never on disk."),
            step: .publishing,
            symbol: "arrow.up.right.circle.fill",
            symbolTint: Brand.accentChrome,
            onBack: { orchestrator.cancelPublish() }
        ) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    identitySection
                    destinationSection
                    readmeSection
                    if let progress = orchestrator.publishProgress {
                        progressCard(progress: progress)
                    }
                    if let failure = orchestrator.publishFailure {
                        ForgeFailureBanner(message: failure)
                    }
                    Spacer(minLength: 0)
                }
            }
            .onAppear { loadFromState() }
        } footer: {
            ForgePrimaryButton(
                orchestrator.isPublishing ? tr("Cancel upload") : tr("Publish"),
                icon: orchestrator.isPublishing ? "xmark" : "arrow.up.circle.fill",
                isEnabled: canPublishOrCancel
            ) {
                if orchestrator.isPublishing {
                    orchestrator.cancelPublish()
                } else {
                    commitAndPublish()
                }
            }
        }
    }

    // MARK: - Sections

    @ViewBuilder
    private var identitySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(tr("IDENTITY"))
            if hasSavedToken && token.isEmpty {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.shield.fill")
                        .foregroundStyle(Brand.success)
                    Text(tr("Using saved Hugging Face token"))
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Brand.typeBody)
                    Spacer(minLength: 0)
                    Button(tr("Replace")) { hasSavedToken = false }
                        .font(.caption2)
                        .buttonStyle(.plain)
                        .foregroundStyle(Brand.accentChrome)
                    Button(tr("Delete")) {
                        orchestrator.deleteHFToken()
                        hasSavedToken = false
                    }
                    .font(.caption2)
                    .buttonStyle(.plain)
                    .foregroundStyle(Brand.danger)
                }
            } else {
                SecureField("hf_…", text: $token)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Brand.typeHi)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(textFieldBackground(focused: !token.isEmpty))
                Text(tr("Need a token? Get one at huggingface.co/settings/tokens (with write access)."))
                    .font(.caption2)
                    .foregroundStyle(Brand.typeTertiary)
            }
            TextField(tr("HF handle (e.g. youssofal)"), text: $handle)
                .textFieldStyle(.plain)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(Brand.typeHi)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(textFieldBackground(focused: !handle.isEmpty))
                .onChange(of: handle) { _, _ in maybeAutoFillRepo() }
        }
    }

    @ViewBuilder
    private var destinationSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(tr("DESTINATION"))
            TextField("owner/repository", text: $repoName)
                .textFieldStyle(.plain)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(Brand.typeHi)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(textFieldBackground(focused: !repoName.isEmpty))
            HStack(spacing: 16) {
                visibilityToggle
                Spacer(minLength: 0)
                licensePicker
            }
        }
    }

    private var visibilityToggle: some View {
        HStack(spacing: 6) {
            ForEach(ForgePublishOptions.Visibility.allCases, id: \.self) { v in
                Button {
                    visibility = v
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: visibility == v ? "largecircle.fill.circle" : "circle")
                            .font(.system(size: 11, weight: .semibold))
                        Text(v.rawValue.capitalized)
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(visibility == v ? Brand.typeHi : Brand.typeSecondary)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var licensePicker: some View {
        Picker(tr("License"), selection: $license) {
            ForEach(licenses, id: \.spdx) { entry in
                Text(entry.label).tag(entry.spdx)
            }
        }
        .pickerStyle(.menu)
        .frame(maxWidth: 220)
    }

    @ViewBuilder
    private var readmeSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                sectionHeader(tr("README.MD PREVIEW"))
                Spacer()
                Button(readmeEditing ? tr("Preview") : tr("Edit")) {
                    readmeEditing.toggle()
                }
                .font(.caption2)
                .buttonStyle(.plain)
                .foregroundStyle(Brand.typeSecondary)
            }
            if readmeEditing {
                TextEditor(text: $readme)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Brand.typeHi)
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .frame(minHeight: 180)
                    .background(textFieldBackground(focused: true))
            } else {
                ScrollView {
                    Text(readme.isEmpty ? generateDefaultREADME() : readme)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Brand.typeBody)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                }
                .frame(minHeight: 180, maxHeight: 240)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Brand.bgInner.opacity(0.6))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .stroke(Brand.separator, lineWidth: 0.5)
                        )
                )
            }
        }
    }

    @ViewBuilder
    private func progressCard(progress: ForgePhaseProgress) -> some View {
        ForgePhaseCard {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(tr("UPLOADING"))
                        .font(.system(size: 10, weight: .heavy, design: .monospaced))
                        .tracking(1.5)
                        .foregroundStyle(Brand.typeTertiary)
                    Spacer(minLength: 0)
                    Text(progress.label ?? "")
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Brand.typeSecondary)
                }
                ForgeLinearProgressBar(
                    fraction: progress.progress,
                    height: 6,
                    minimumFillWidth: 6
                )
            }
        }
    }

    // MARK: - Helpers

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .heavy, design: .monospaced))
            .tracking(1.5)
            .foregroundStyle(Brand.typeTertiary)
    }

    private func textFieldBackground(focused: Bool) -> some View {
        RoundedRectangle(cornerRadius: 8, style: .continuous)
            .fill(Brand.bgOuter)
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(
                        focused ? Brand.accentChrome.opacity(0.6) : Brand.separator,
                        lineWidth: focused ? 0.75 : 0.5
                    )
            )
    }

    private var canPublishOrCancel: Bool {
        if orchestrator.isPublishing { return true }
        let trimmedRepo = repoName.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmedRepo.isEmpty && (hasSavedToken || !token.isEmpty)
    }

    private func loadFromState() {
        handle = backend.configuration.huggingFaceHandle ?? ""
        repoName = state(\.publish.repoName).isEmpty
            ? defaultRepoName()
            : state(\.publish.repoName)
        visibility = state(\.publish.visibility)
        license = state(\.publish.licenseSPDX).isEmpty ? "apache-2.0" : state(\.publish.licenseSPDX)
        readme = state(\.publish.readmeBody)
        hasSavedToken = orchestrator.savedHFToken != nil
    }

    private func state<V>(_ keyPath: KeyPath<ForgeFeatureState, V>) -> V {
        orchestrator.state[keyPath: keyPath]
    }

    private func defaultRepoName() -> String {
        let trimmedHandle = handle.trimmingCharacters(in: .whitespacesAndNewlines)
        let branded = orchestrator.state.brand.brandedName
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHandle.isEmpty, !branded.isEmpty else { return branded }
        return "\(trimmedHandle)/\(branded)"
    }

    private func maybeAutoFillRepo() {
        let trimmedHandle = handle.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedRepo = repoName.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedRepo.isEmpty || !trimmedRepo.contains("/") {
            repoName = "\(trimmedHandle)/\(orchestrator.state.brand.brandedName)"
        }
    }

    private func commitAndPublish() {
        // 1. Save token to Keychain if the user typed a fresh one.
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedToken.isEmpty {
            _ = orchestrator.saveHFToken(trimmedToken)
        }
        // 2. Persist HF handle in the app config.
        var config = backend.configuration
        let trimmedHandle = handle.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedHandle.isEmpty {
            config.huggingFaceHandle = trimmedHandle
            try? backend.saveSettings(config)
        }
        // 3. Snapshot the publish options into the orchestrator state.
        let options = ForgePublishOptions(
            repoName: repoName.trimmingCharacters(in: .whitespacesAndNewlines),
            visibility: visibility,
            licenseSPDX: license,
            readmeBody: readme.isEmpty ? generateDefaultREADME() : readme
        )
        orchestrator.updatePublishOptions(options)
        orchestrator.startPublish()
    }

    private func generateDefaultREADME() -> String {
        let brand = orchestrator.state.brand.brandedName
        let source = orchestrator.state.sourceProbe?.hfRepo ?? "<source-repo>"
        let depth = orchestrator.state.verification?.bestDepth ?? 0
        let multiplier = orchestrator.state.verification?.multiplierVsAr ?? 1.0
        let hardware = orchestrator.state.hardware?.chipName ?? "Apple Silicon"
        return """
        # \(brand)

        MTPLX-branded multi-token-prediction model for Apple Silicon (MLX).
        Forged with [MTPLX Forge](https://github.com/youssofal/MTPLX) from
        `\(source)`.

        ## Verification

        - Best depth: D\(depth)
        - Multiplier vs autoregressive baseline: \(String(format: "%.2f×", multiplier))
        - Verified on: \(hardware)
        - Sampler: temperature 0.6 · top_p 0.95 · top_k 20

        See `mtplx_runtime.json` for the full verification record.

        ## Usage

        ```bash
        # MTPLX picks this model up automatically when downloaded:
        mtplx pull \(orchestrator.state.publish.repoName.isEmpty ? "<owner>/\(brand)" : orchestrator.state.publish.repoName)
        mtplx start chat
        ```

        ## License

        See LICENSE.
        """
    }
}
