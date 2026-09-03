import SwiftUI
import AppKit
import MTPLXAppCore

// MARK: - AboutSheet
//
// Brand-led About panel. Wordmark + tagline + version metadata +
// configuration overview. Phase 8.7 flesh-out adds capabilities,
// endpoints, mutable-settings list, and feature-flag table.

struct AboutSheet: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                heroSection
                appSection
                updateSection
                runtimeSection
                modelsSection
                if let caps = backend.capabilities {
                    capabilitiesSection(caps)
                    endpointsSection(caps)
                    featuresSection(caps)
                    mutableSettingsSection(caps)
                }
                connectionSection
                settingsLocationSection
            }
            .padding(28)
        }
        .frame(minWidth: 540, minHeight: 560)
        .background(Brand.pianoRadial.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .tint(Brand.accent)
        .toolbar { dismissButton }
        .task {
            await backend.refreshModels()
            await backend.refreshRuntimeUpdateStatus()
        }
    }

    // MARK: - Sections

    @ViewBuilder
    private var heroSection: some View {
        HStack(alignment: .top, spacing: 24) {
            WordmarkView(height: 40)
            Spacer()
            Button(tr("Close")) { dismiss() }
                .buttonStyle(.bordered)
                .keyboardShortcut(.cancelAction)
        }
        WordmarkSubtitle(dividerWidth: 280)
        Text(tr("Fast local AI for Apple Silicon."))
            .font(.system(.callout, design: .monospaced))
            .foregroundStyle(Brand.textHighlight)
            .padding(.top, 4)
    }

    @ViewBuilder
    private var appSection: some View {
        sectionHeader(tr("APP"))
        VStack(alignment: .leading, spacing: 6) {
            row(tr("Version"), value: appVersion)
            row(tr("Build"), value: appBuild)
            row(tr("Bundle ID"), value: bundleIdentifier)
            row(tr("Bundle path"), value: Bundle.main.bundleURL.path)
        }
    }

    @ViewBuilder
    private var runtimeSection: some View {
        let health = backend.health
        sectionHeader(tr("RUNTIME"))
        VStack(alignment: .leading, spacing: 6) {
            row(tr("Model"), value: health?.model ?? "—")
            row(tr("Generation"), value: (health?.generationMode ?? "—").uppercased())
            row(tr("MTP depth"), value: health.map { tr("D%lld", $0.depth) } ?? "—")
            row(tr("Context window"), value: Format.integer(health?.contextWindow))
        }
    }

    @ViewBuilder
    private var updateSection: some View {
        sectionHeader(tr("UPDATES"))
        VStack(alignment: .leading, spacing: 6) {
            if let snapshot = backend.runtimeUpdateSnapshot {
                row(tr("Latest app"), value: snapshot.latestAppVersion ?? "—")
                row(tr("CLI version"), value: snapshot.cliVersion ?? "—")
                row(tr("CLI path"), value: snapshot.cliPath ?? "—")
                row(tr("CLI install"), value: snapshot.cliInstallKind.displayName)
                row(tr("CLI latest"), value: snapshot.recommendedCLIVersion ?? "—")
                row(snapshot.title, value: snapshot.detail)
                HStack(spacing: 10) {
                    Button {
                        Task { await backend.refreshRuntimeUpdateStatus() }
                    } label: {
                        Label(tr("Check"), systemImage: "arrow.clockwise")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    if snapshot.canUpdateRuntime {
                        Button {
                            Task { await backend.updateRuntimeWithHomebrew() }
                        } label: {
                            Label(tr("Update Runtime"), systemImage: "arrow.down.circle")
                                .font(.system(size: 11, weight: .medium))
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }
                    Spacer()
                }
                if let failure = backend.runtimeUpdateFailure {
                    Text(failure)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(Brand.warning)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else {
                row(tr("Status"), value: "Checking...")
            }
        }
    }

    @ViewBuilder
    private var modelsSection: some View {
        if let models = backend.models, !models.data.isEmpty {
            sectionHeader(tr("MODELS"))
            VStack(alignment: .leading, spacing: 6) {
                ForEach(models.data) { model in
                    row(model.id, value: model.ownedBy ?? "—")
                }
            }
        }
    }

    @ViewBuilder
    private func capabilitiesSection(_ caps: AppCapabilities) -> some View {
        sectionHeader(tr("CAPABILITIES"))
        VStack(alignment: .leading, spacing: 6) {
            row(tr("API version"), value: String(caps.apiVersion))
            row(tr("Endpoint name"), value: caps.name)
            row(
                tr("Snapshot interval"),
                value: tr("%lld–%lld ms (default %lld)", caps.snapshotInterval.minMs, caps.snapshotInterval.maxMs, caps.snapshotInterval.defaultMs)
            )
            row(
                tr("Performance Lock cadence"),
                value: tr("%lld ms", caps.snapshotInterval.performanceLockMs)
            )
        }
    }

    @ViewBuilder
    private func endpointsSection(_ caps: AppCapabilities) -> some View {
        if !caps.endpoints.isEmpty {
            sectionHeader(tr("ENDPOINTS"))
            VStack(alignment: .leading, spacing: 6) {
                ForEach(caps.endpoints.sorted(by: { $0.key < $1.key }), id: \.key) { entry in
                    row(entry.key, value: entry.value)
                }
            }
        }
    }

    @ViewBuilder
    private func featuresSection(_ caps: AppCapabilities) -> some View {
        if !caps.features.isEmpty {
            sectionHeader(tr("FEATURE FLAGS"))
            VStack(alignment: .leading, spacing: 6) {
                ForEach(caps.features.sorted(by: { $0.key < $1.key }), id: \.key) { entry in
                    HStack {
                        Text(entry.key)
                            .font(.system(.callout, design: .monospaced))
                            .foregroundStyle(Brand.textHighlight.opacity(0.7))
                        Spacer()
                        Image(systemName: entry.value ? "checkmark.circle.fill" : "minus.circle")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(entry.value ? Brand.success : Brand.textHighlight.opacity(0.4))
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func mutableSettingsSection(_ caps: AppCapabilities) -> some View {
        if !caps.mutableSettings.isEmpty || !caps.restartRequiredSettings.isEmpty {
            sectionHeader(tr("SETTINGS POLICY"))
            VStack(alignment: .leading, spacing: 6) {
                if !caps.mutableSettings.isEmpty {
                    row(tr("Live mutable"), value: caps.mutableSettings.joined(separator: ", "))
                }
                if !caps.restartRequiredSettings.isEmpty {
                    row(tr("Restart required"), value: caps.restartRequiredSettings.joined(separator: ", "))
                }
            }
        }
    }

    @ViewBuilder
    private var connectionSection: some View {
        sectionHeader(tr("CONNECTION"))
        VStack(alignment: .leading, spacing: 6) {
            row(tr("Endpoint"), value: "\(backend.configuration.host):\(backend.configuration.port)")
            row(tr("Stream cadence"), value: cadenceText)
        }
    }

    @ViewBuilder
    private var settingsLocationSection: some View {
        sectionHeader(tr("APP SETTINGS"))
        VStack(alignment: .leading, spacing: 6) {
            row(tr("Settings file"), value: settingsPath)
            HStack {
                Spacer()
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: settingsPath)])
                } label: {
                    Label(tr("Reveal in Finder"), systemImage: "folder")
                        .font(.system(size: 11, weight: .medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
    }

    private var settingsPath: String {
        backend.settingsURL.path
    }

    private var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown"
    }

    private var appBuild: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "unknown"
    }

    private var bundleIdentifier: String {
        Bundle.main.bundleIdentifier ?? "unknown"
    }

    private var cadenceText: String {
        let config = backend.configuration
        if config.performanceLock {
            return tr("1000 ms (Performance Lock)")
        }
        return tr("%lld ms", config.streamSnapshotIntervalMs)
    }

    // MARK: - Helpers

    @ToolbarContentBuilder
    private var dismissButton: some ToolbarContent {
        ToolbarItem(placement: .cancellationAction) {
            Button(tr("Close")) { dismiss() }
                .keyboardShortcut(.cancelAction)
        }
    }

    @ViewBuilder
    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 11, weight: .heavy, design: .monospaced))
            .tracking(3)
            .foregroundStyle(Brand.textHighlight.opacity(0.6))
    }

    @ViewBuilder
    private func row(_ label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(Brand.textHighlight.opacity(0.65))
            Spacer(minLength: 12)
            Text(value)
                .font(.system(.callout, design: .monospaced).weight(.medium))
                .foregroundStyle(Brand.accent)
                .lineLimit(2)
                .truncationMode(.middle)
                .multilineTextAlignment(.trailing)
        }
    }
}
