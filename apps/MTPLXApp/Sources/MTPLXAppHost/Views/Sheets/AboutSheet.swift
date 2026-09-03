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
        .appliesAppearance()
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
            Button("Close") { dismiss() }
                .buttonStyle(.bordered)
                .keyboardShortcut(.cancelAction)
        }
        WordmarkSubtitle(dividerWidth: 280)
        Text("Fast local AI for Apple Silicon.")
            .font(.system(.callout, design: .monospaced))
            .foregroundStyle(Brand.textHighlight)
            .padding(.top, 4)
    }

    @ViewBuilder
    private var appSection: some View {
        sectionHeader("APP")
        VStack(alignment: .leading, spacing: 6) {
            row("Version", value: appVersion)
            row("Build", value: appBuild)
            row("Bundle ID", value: bundleIdentifier)
            row("Bundle path", value: Bundle.main.bundleURL.path)
        }
    }

    @ViewBuilder
    private var runtimeSection: some View {
        let health = backend.health
        sectionHeader("RUNTIME")
        VStack(alignment: .leading, spacing: 6) {
            row("Model", value: health?.model ?? "—")
            row("Generation", value: (health?.generationMode ?? "—").uppercased())
            row("MTP depth", value: health.map { "D\($0.depth)" } ?? "—")
            row("Context window", value: Format.integer(health?.contextWindow))
        }
    }

    @ViewBuilder
    private var updateSection: some View {
        sectionHeader("UPDATES")
        VStack(alignment: .leading, spacing: 6) {
            if let snapshot = backend.runtimeUpdateSnapshot {
                row("Latest app", value: snapshot.latestAppVersion ?? "—")
                row("CLI version", value: snapshot.cliVersion ?? "—")
                row("CLI path", value: snapshot.cliPath ?? "—")
                row("CLI install", value: snapshot.cliInstallKind.displayName)
                row("CLI latest", value: snapshot.recommendedCLIVersion ?? "—")
                row(snapshot.title, value: snapshot.detail)
                HStack(spacing: 10) {
                    Button {
                        Task { await backend.refreshRuntimeUpdateStatus() }
                    } label: {
                        Label("Check", systemImage: "arrow.clockwise")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    if snapshot.canUpdateRuntime {
                        Button {
                            Task { await backend.updateRuntimeWithHomebrew() }
                        } label: {
                            Label("Update Runtime", systemImage: "arrow.down.circle")
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
                row("Status", value: "Checking...")
            }
        }
    }

    @ViewBuilder
    private var modelsSection: some View {
        if let models = backend.models, !models.data.isEmpty {
            sectionHeader("MODELS")
            VStack(alignment: .leading, spacing: 6) {
                ForEach(models.data) { model in
                    row(model.id, value: model.ownedBy ?? "—")
                }
            }
        }
    }

    @ViewBuilder
    private func capabilitiesSection(_ caps: AppCapabilities) -> some View {
        sectionHeader("CAPABILITIES")
        VStack(alignment: .leading, spacing: 6) {
            row("API version", value: String(caps.apiVersion))
            row("Endpoint name", value: caps.name)
            row(
                "Snapshot interval",
                value: "\(caps.snapshotInterval.minMs)–\(caps.snapshotInterval.maxMs) ms (default \(caps.snapshotInterval.defaultMs))"
            )
            row(
                "Performance Lock cadence",
                value: "\(caps.snapshotInterval.performanceLockMs) ms"
            )
        }
    }

    @ViewBuilder
    private func endpointsSection(_ caps: AppCapabilities) -> some View {
        if !caps.endpoints.isEmpty {
            sectionHeader("ENDPOINTS")
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
            sectionHeader("FEATURE FLAGS")
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
            sectionHeader("SETTINGS POLICY")
            VStack(alignment: .leading, spacing: 6) {
                if !caps.mutableSettings.isEmpty {
                    row("Live mutable", value: caps.mutableSettings.joined(separator: ", "))
                }
                if !caps.restartRequiredSettings.isEmpty {
                    row("Restart required", value: caps.restartRequiredSettings.joined(separator: ", "))
                }
            }
        }
    }

    @ViewBuilder
    private var connectionSection: some View {
        sectionHeader("CONNECTION")
        VStack(alignment: .leading, spacing: 6) {
            row("Endpoint", value: "\(backend.configuration.host):\(backend.configuration.port)")
            row("Stream cadence", value: cadenceText)
        }
    }

    @ViewBuilder
    private var settingsLocationSection: some View {
        sectionHeader("APP SETTINGS")
        VStack(alignment: .leading, spacing: 6) {
            row("Settings file", value: settingsPath)
            HStack {
                Spacer()
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: settingsPath)])
                } label: {
                    Label("Reveal in Finder", systemImage: "folder")
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
            return "1000 ms (Performance Lock)"
        }
        return "\(config.streamSnapshotIntervalMs) ms"
    }

    // MARK: - Helpers

    @ToolbarContentBuilder
    private var dismissButton: some ToolbarContent {
        ToolbarItem(placement: .cancellationAction) {
            Button("Close") { dismiss() }
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
